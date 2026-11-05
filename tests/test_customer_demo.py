"""Customer journey and policy isolation without upstream costs or downloads."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sdk/python/src'))
from fastapi.testclient import TestClient
from lonn.demo.api import create_app
from lonn.demo.store import Store
from lonn.schemas import ModelResponse, TokenUsage
from lonnrouter import LonnRouter, AsyncLonnRouter
import httpx


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.app = create_app(self.temp.name)
        self.client = TestClient(self.app).__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post('/api/demo/login', json={'email':'admin@nordic.demo','password':'LonnDemo2026!'})
        self.account = self.client.get('/api/account').json()
        self.headers = {'Authorization': 'Bearer ' + self.account['api_key'], 'X-Lonn-Application': 'engineering-app'}
        self.request = {'prompt': 'Solve 2 + 2', 'signal': {'task_scores': {'math_problem': 1.0}}, 'verify': False}

    def generate(self, **kwargs):
        return self.client.post('/generate', headers=self.headers, json=dict(self.request, **kwargs))

    @staticmethod
    def answer(prompt, model, **kwargs):
        return ModelResponse('4', model, TokenUsage(10, 20, 30, .00001, .00002, .00003), 20, 'stop')

    def credential(self):
        result = self.client.put('/api/provider', json={'key': 'upstream-secret-for-test'})
        self.assertEqual(result.status_code, 200, result.text)

    def test_account_seed_persistence_reset(self):
        self.assertEqual(len(self.client.get('/api/usage').json()), 64)
        settings = dict(self.account['settings'], lam=0.71)
        self.assertEqual(self.client.put('/api/settings', json=settings).status_code, 200)
        reopened = Store(self.temp.name)
        self.assertEqual(reopened.settings()['lam'], 0.71)
        self.assertEqual(reopened.account().api_key, self.account['api_key'])
        self.credential()
        reopened.reset()
        self.assertEqual(len(reopened.history()), 64)
        self.assertEqual(reopened.settings()['lam'], 0.2)
        self.assertNotEqual(reopened.account().api_key, self.account['api_key'])
        self.assertEqual(reopened.credential(), 'upstream-secret-for-test')
        reopened.engine.dispose()

    def test_seed_is_reproducible_varied_and_refresh_preserves_live_usage(self):
        from collections import Counter
        original = self.client.get('/api/usage').json()
        self.assertGreater(len(set(r['input_tokens'] + r['output_tokens'] for r in original)), 50)
        distribution = Counter(r['model'] for r in original)
        self.assertGreater(max(distribution.values()), min(distribution.values()))
        self.app.state.store.refresh_seeded_history()
        self.assertEqual(original, self.client.get('/api/usage').json())
        self.generate()  # Record a live missing-credential failure.
        self.app.state.store.refresh_seeded_history()
        self.assertEqual(len(self.client.get('/api/usage').json()), 65)

    def test_tariff_migration_refreshes_only_synthetic_records(self):
        from sqlmodel import Session
        from lonn.demo.store import Usage
        self.credential()
        with patch('lonn.demo.api.execute', side_effect=self.answer):
            self.generate()
        live = next(r for r in self.client.get('/api/usage').json() if r['source'] == 'live')
        # Simulate a real invoice from the former tariff and an old synthetic fixture.
        with Session(self.app.state.store.engine) as session:
            for row_id in (live['id'], 'seed-000'):
                row = session.get(Usage, row_id)
                detail = json.loads(row.details)
                detail['billing'].pop('service_fee_basis')
                row.details = json.dumps(detail)
                row.cost_usd = .123456
                session.add(row)
            session.commit()
        historical = next(r for r in self.client.get('/api/usage').json() if r['id'] == live['id'])
        reopened = Store(self.temp.name)
        after = self.client.get('/api/usage').json()
        self.assertEqual(next(r for r in after if r['id'] == live['id']), historical)
        self.assertEqual(next(r for r in after if r['id'] == 'seed-000')['details']['billing']['service_fee_basis'], 'estimated_savings_v1')
        self.assertEqual(len(after), 65)
        self.assertEqual(reopened.account().api_key, self.account['api_key'])
        self.assertEqual(reopened.credential(), 'upstream-secret-for-test')
        reopened.engine.dispose()

    def test_seeded_models_and_costs_follow_recorded_routing_signals(self):
        from lonn.demo.pricing import charge
        from lonn.schemas import TaskSignal
        from lonn.task_router import rank_models
        pool = self.app.state.models
        table = self.app.state.table
        for row in self.client.get('/api/usage').json():
            signal = TaskSignal(row['details']['signal']['task_scores'])
            ranked = rank_models(signal, table, pool, lam=0.2)
            expected = ranked[0] if row['status'] == 'success' else ranked[-1]
            self.assertEqual(row['model'], expected.id)
            self.assertEqual(row['details']['lam'], 0.2)
            bill = charge(row['input_tokens'], row['output_tokens'], expected.cost_per_m_input, expected.cost_per_m_output,
                          model_id=expected.id, top_model_id=row['details']['baseline']['model_id'])
            self.assertAlmostEqual(row['cost_usd'], bill['total_cost_usd'])
            self.assertEqual(row['details']['billing'], bill)

    def test_invalid_provider_input_never_echoes_credential(self):
        secret = 'secret-' * 100
        response = self.client.put('/api/provider', json={'key': secret})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(secret, response.text)

    def test_authentication_origin_and_rotation(self):
        self.assertEqual(self.client.post('/generate', json=self.request).status_code, 401)
        self.assertEqual(self.client.get('/models').status_code, 401)
        self.assertEqual(self.client.post('/api/demo/login', headers={'Origin': 'https://evil.example'}).status_code, 403)
        new = self.client.post('/api/key/rotate').json()['api_key']
        self.assertNotEqual(new, self.account['api_key'])
        self.assertEqual(self.generate().status_code, 401)
        self.client.post('/api/demo/logout')
        self.assertEqual(self.client.get('/api/account').status_code, 401)

    def test_settings_override_sdk_lambda_and_unknown_privacy_fails_closed(self):
        for name, value in [('require_zdr', True), ('forbid_training', True), ('region', 'EU'), ('max_retention_days', 0)]:
            settings = dict(self.account['settings'], **{name: value})
            self.client.put('/api/settings', json=settings)
            with patch('lonn.demo.api.execute') as execute:
                response = self.generate(lam=0)
                self.assertEqual(response.status_code, 422, response.text)
                execute.assert_not_called()
            self.assertFalse(any(m['eligible'] for m in self.client.get('/api/catalog').json()))
        self.client.put('/api/settings', json=dict(self.account['settings'], lam=0.85, enforce_lam=True))
        result = self.client.post('/route', headers=self.headers, json={k:v for k,v in self.request.items() if k!='verify'}).json()
        self.assertEqual(result['lam'], 0.85)

    def test_full_sdk_journey_and_no_credential_or_prompt_in_telemetry(self):
        self.credential()
        with patch('lonn.demo.api.execute', side_effect=self.answer) as execute:
            with LonnRouter(base_url='http://testserver', api_key=self.account['api_key'], application='engineering-app', http_client=self.client) as sdk:
                result = sdk.generate('Solve 2 + 2', signal=self.request['signal'], verify=False)
                self.assertEqual(result.content, '4')
                self.assertEqual(result.billing.service_fee_basis, 'estimated_savings_v1')
                self.assertEqual(result.total_cost_usd, result.billing.total_cost_usd)
            self.assertEqual(execute.call_args.kwargs['api_key'], 'upstream-secret-for-test')
        history = self.client.get('/api/usage').json()
        live = [r for r in history if r['source']=='live']
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0]['input_tokens'], 10)
        self.assertEqual(live[0]['status'], 'success')
        for path in ['/api/account','/api/usage','/api/catalog','/openapi.json']:
            self.assertNotIn('upstream-secret-for-test', self.client.get(path).text)
        self.assertNotIn('Solve 2 + 2', json.dumps(history))
        self.assertEqual((Path(self.temp.name)/'provider.key').stat().st_mode & 0o777, 0o600)

    def test_disabled_models_excluded_from_escalation_and_all_attempt_tokens_counted(self):
        self.credential()
        catalog = self.client.get('/api/catalog').json()
        eligible = [m['id'] for m in catalog if m['eligible']]
        allowed = eligible[:2]
        self.client.put('/api/settings', json=dict(self.account['settings'], lam=0.0,
                        disabled_models=[m for m in eligible if m not in allowed]))
        def truncated(prompt, model, **kwargs):
            self.assertIn(model.id, allowed)
            return ModelResponse('', model, TokenUsage(10,20,30,.1,.2,.3), 20, 'length')
        with patch('lonn.demo.api.execute', side_effect=truncated) as execute:
            result = self.generate(verify=True)
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(execute.call_count, 2)
        row = next(r for r in self.client.get('/api/usage').json() if r['source']=='live')
        self.assertEqual(row['input_tokens'], 20)
        self.assertEqual(row['output_tokens'], 40)
        from lonn.demo.pricing import request_billing, token_cost
        expected = request_billing([(m.id, token_cost(10, 20, m.cost_per_m_input, m.cost_per_m_output))
                                    for m in self.app.state.models if m.id in allowed], row['details']['baseline']['model_id'])
        self.assertEqual(row['details']['billing'], expected)
        self.assertAlmostEqual(row['cost_usd'], expected['total_cost_usd'])
        self.assertEqual(result.json()['billing'], expected)
        self.assertAlmostEqual(sum(a['cost_usd'] for a in result.json()['attempts']), expected['total_cost_usd'])
        self.assertAlmostEqual(sum(a['service_fee'] for a in row['details']['attempt_costs']), expected['service_fee_usd'])
        self.assertTrue(row['details']['escalated'])

    def test_failed_provider_is_sanitized_and_recorded(self):
        self.credential()
        with patch('lonn.demo.api.execute', side_effect=RuntimeError('upstream-secret-for-test')):
            response = self.generate()
        self.assertEqual(response.status_code, 502)
        self.assertNotIn('upstream-secret-for-test', response.text)
        row = next(r for r in self.client.get('/api/usage').json() if r['source']=='live')
        self.assertEqual(row['status'], 'failed')
        self.assertFalse(row['details']['usage_complete'])

    def test_missing_provider_recorded_without_execution(self):
        with patch('lonn.demo.api.execute') as execute:
            self.assertEqual(self.generate().status_code, 503)
            execute.assert_not_called()
        row = next(r for r in self.client.get('/api/usage').json() if r['source']=='live')
        self.assertEqual(row['status'], 'failed')

    def test_customer_fee_is_twenty_percent_of_positive_benchmark_savings(self):
        from lonn.demo.pricing import BENCHMARK_COSTS, charge, request_billing
        selected = 'openrouter/openai/gpt-oss-20b'
        top = 'openrouter/anthropic/claude-sonnet-5.5'
        bill = charge(1_000_000, 500_000, 2, 10, model_id=selected, top_model_id=top)
        self.assertEqual(bill['model_cost_usd'], 7)
        self.assertAlmostEqual(bill['estimated_top_model_cost_usd'], 7 * 7.57 / .19)
        self.assertAlmostEqual(bill['service_fee_usd'], .2 * (7 * 7.57 / .19 - 7))
        self.assertAlmostEqual(bill['total_cost_usd'], 7 + bill['service_fee_usd'])
        for costs, baseline in [([(top, 7)], top), ([(top, 7)], selected), ([(selected, 0)], top)]:
            self.assertEqual(request_billing(costs, baseline)['service_fee_usd'], 0)
        # Clamp once after summing: positive savings on one attempt cannot hide an overall loss.
        mixed = request_billing([(selected, .001), (top, 10)], 'openrouter/meta/muse-spark-1.3')
        self.assertLess(mixed['estimated_savings_usd'], 0)
        self.assertEqual(mixed['service_fee_usd'], 0)
        for costs, baseline in [([('unknown', 7)], top), ([(selected, 7)], 'unknown')]:
            unknown = request_billing(costs, baseline)
            self.assertIsNone(unknown['estimated_top_model_cost_usd'])
            self.assertEqual(unknown['service_fee_usd'], 0)
            self.assertEqual(unknown['total_cost_usd'], 7)
        self.credential()
        def billed(prompt, model, **kwargs):
            return ModelResponse('4', model, TokenUsage(10, 20, 30, .1, .2, 99,
                                 'provider_reported', provider_cost=99), 15, 'stop')
        with patch('lonn.demo.api.execute', side_effect=billed):
            result = self.generate(model=selected).json()
        row = next(r for r in self.client.get('/api/usage').json() if r['source'] == 'live')
        self.assertNotEqual(row['details']['baseline']['model_id'], selected)
        model = result['model']
        expected = charge(10, 20, model['cost_per_m_input'], model['cost_per_m_output'],
                          model_id=selected, top_model_id=row['details']['baseline']['model_id'])
        self.assertEqual(result['billing'], expected)
        self.assertEqual(row['details']['billing'], expected)
        self.assertEqual(result['total_cost_usd'], expected['total_cost_usd'])
        self.assertEqual(result['cost_basis'], 'calculated')
        self.assertEqual(result['response']['usage']['provider_cost'], 99)
        self.assertEqual(result['response']['usage']['service_fee'], expected['service_fee_usd'])
        # Missing benchmark data never invents a fee, even for otherwise valid requests.
        with patch.dict(BENCHMARK_COSTS, {}, clear=True), patch('lonn.demo.api.execute', side_effect=billed):
            unknown = self.generate(model=selected).json()
        self.assertIsNone(unknown['billing']['estimated_top_model_cost_usd'])
        self.assertEqual(unknown['billing']['service_fee_usd'], 0)
        self.assertEqual(unknown['total_cost_usd'], unknown['billing']['model_cost_usd'])

    def login_as(self, email):
        self.client.post('/api/demo/logout')
        response = self.client.post('/api/demo/login', json={'email':email,'password':'LonnDemo2026!'})
        self.assertEqual(response.status_code, 200, response.text)

    def test_application_key_enforces_team_policy_rotation_and_attribution(self):
        self.credential()
        team = self.client.post('/api/teams', json={'name':'Research'}).json()
        app_id='app-'+team['id']
        application=next(a for a in self.client.get('/api/hierarchy').json()['applications'] if a['id']==app_id)
        models = [m for m in self.client.get('/api/catalog').json() if m['eligible']]
        allow = models[0]['id']
        policy = {'name':team['name'],'enabled':True,'disabled_models':[m['id'] for m in models[1:]]}
        self.assertEqual(self.client.put('/api/teams/'+team['id'],json=policy).status_code,200)
        headers = {'Authorization':'Bearer '+application['api_key']}
        with patch('lonn.demo.api.execute',side_effect=self.answer) as execute:
            result = self.client.post('/generate',headers=headers,json=dict(self.request,verify=True))
            self.assertEqual(result.status_code,200,result.text)
            self.assertTrue(all(c.args[1].id == allow for c in execute.call_args_list))
        row=next(r for r in self.client.get('/api/usage').json() if r['source']=='live')
        self.assertEqual((row['team_id'],row['application_id'],row['user_id']),(team['id'],app_id,None))
        self.client.put('/api/settings',json=dict(self.account['settings'],disabled_models=[allow]))
        self.assertEqual(self.client.post('/generate',headers=headers,json=self.request).status_code,422)
        rotated=self.client.post('/api/applications/'+app_id+'/rotate').json()
        self.assertNotEqual(application['api_key'],rotated['api_key'])
        self.assertEqual(self.client.get('/models',headers=headers).status_code,401)
        self.client.put('/api/teams/'+team['id'],json=dict(policy,enabled=False))
        self.assertEqual(self.client.get('/models',headers={'Authorization':'Bearer '+rotated['api_key']}).status_code,401)

    def test_role_permissions_and_private_account_isolation(self):
        self.login_as('robin@nordic.demo')
        member=self.client.get('/api/account').json()
        self.assertEqual(member['user']['role'],'enterprise_user')
        self.assertNotEqual(member['api_key'],self.account['api_key'])
        self.assertTrue(all(r['team_id']=='team-engineering' for r in self.client.get('/api/usage').json()))
        for method,path,body in [('put','/api/settings',self.account['settings']),
                                ('put','/api/provider',{'key':'forbidden'}),
                                ('post','/api/teams',{'name':'forbidden'}),
                                ('post','/api/teams',{'name':'forbidden'})]:
            self.assertEqual(getattr(self.client,method)(path,json=body).status_code,403)
        self.assertEqual(self.client.get('/api/users').status_code,403)
        self.login_as('sam@personal.demo')
        personal=self.client.get('/api/account').json()
        self.assertEqual(personal['id'],'demo-personal')
        self.assertEqual(self.client.get('/api/usage').json(),[])
        self.assertEqual(len(self.client.get('/api/teams').json()),1)
        self.assertEqual(self.client.put('/api/settings',json=dict(personal['settings'],lam=0.31)).status_code,200)
        self.assertEqual(self.app.state.store.settings()['lam'],0.2)
        self.assertEqual(self.client.post('/generate',headers={'Authorization':'Bearer '+personal['api_key'], 'X-Lonn-Application':'personal-app'},json=self.request).status_code,503)
        self.assertEqual(len(self.client.get('/api/usage').json()),1)
        self.login_as('admin@nordic.demo')
        self.assertEqual(len(self.client.get('/api/usage').json()),64)

    def test_user_creation_password_hashing_membership_and_disabled_sessions(self):
        response=self.client.post('/api/users',json={'name':'Taylor','email':'taylor@nordic.demo',
            'password':'LocalDemoPassword!','team_id':'team-support','role':'enterprise_admin'})
        self.assertEqual(response.status_code,201,response.text)
        user=response.json()
        self.assertEqual(user['role'],'enterprise_user')
        self.assertNotIn('password',response.text)
        saved=self.app.state.store.user(user_id=user['id'])
        self.assertNotIn('LocalDemoPassword!',saved.password_hash)
        self.assertEqual(self.client.post('/api/demo/login',json={'email':user['email'],'password':'wrong'}).status_code,401)
        other=TestClient(self.app)  # Reuse running app state; no second lifespan.
        self.assertEqual(other.post('/api/demo/login',json={'email':user['email'],'password':'LocalDemoPassword!'}).status_code,200)
        self.assertEqual(other.get('/api/account').json()['user']['team_id'],'team-support')
        self.client.put('/api/users/'+user['id'],json={'team_id':'team-content','enabled':True})
        self.assertEqual(other.get('/api/account').json()['user']['team_id'],'team-content')
        self.client.put('/api/users/'+user['id'],json={'team_id':'team-content','enabled':False})
        self.assertEqual(other.get('/api/account').status_code,401)
        other.close()

    def test_baseline_uses_quality_and_preserves_historical_rates_and_pool(self):
        from lonn.demo.baseline import top_model_baseline
        from lonn.routing_table import build_table
        from lonn.schemas import Model, TaskSignal
        strong = Model('openrouter/test/strong','openrouter',1,2,8192,{})
        pricey = Model('openrouter/test/pricey','openrouter',100,200,8192,{})
        table = build_table([{'task':'code_function','difficulty':None,'model_id':m.id,
                            'correct':ok,'cost_usd':cost} for m,ok,cost in
                            [(strong,True,.01),(pricey,False,1)] for _ in range(10)])
        baseline=top_model_baseline([pricey,strong],table,1000,500)
        self.assertEqual(baseline['model_id'],strong.id)
        self.assertEqual(baseline['input_rate_per_m'], 1)
        self.assertEqual(baseline['output_rate_per_m'], 2)
        self.assertIsNone(baseline['benchmark_cost_per_690'])
        self.assertEqual(top_model_baseline([pricey],table,1000,500)['model_id'],pricey.id)
        self.credential()
        with patch('lonn.demo.api.execute',side_effect=self.answer):
            self.assertEqual(self.generate().status_code,200)
        row=next(r for r in self.client.get('/api/usage').json() if r['source']=='live')
        saved=row['details']['baseline']
        self.assertEqual(saved['estimated_cost_usd'], row['details']['billing']['estimated_top_model_cost_usd'])
        self.assertEqual(saved['input_tokens'],row['input_tokens'])
        self.assertEqual(saved['output_tokens'],row['output_tokens'])
        self.client.put('/api/settings',json=dict(self.account['settings'],disabled_models=[saved['model_id']]))
        after=next(r for r in self.client.get('/api/usage').json() if r['id']==row['id'])
        self.assertEqual(after['details']['baseline'],saved)

    def test_pool_baseline_uses_overall_quality_not_task_specific_winner(self):
        from lonn.demo.baseline import top_model_baseline
        from lonn.routing_table import build_table
        from lonn.schemas import Model, TaskSignal
        from lonn.task_router import rank_models
        strong = Model('openrouter/test/strong','openrouter',1,2,8192,{})
        specialist = Model('openrouter/test/specialist','openrouter',.1,.2,8192,{})
        rows = []
        for model, task, passes in [(strong,'code_function',4),(strong,'math_problem',10),
                                     (specialist,'code_function',8),(specialist,'math_problem',4)]:
            rows.extend({'task':task,'difficulty':None,'model_id':model.id,
                         'correct':i < passes,'cost_usd':.001} for i in range(10))
        table = build_table(rows)
        self.assertEqual(rank_models(TaskSignal({'code_function':1.0}),table,[strong,specialist],lam=0)[0].id,specialist.id)
        baseline = top_model_baseline([strong,specialist],table,10,20)
        self.assertEqual(baseline['model_id'],strong.id)
        self.assertEqual(baseline['ranking'],'global_benchmark_v1')
        self.assertEqual(set(baseline['eligible_model_ids']),{strong.id,specialist.id})
        self.assertEqual(top_model_baseline([specialist],table,10,20)['model_id'],specialist.id)
        self.assertIsNone(top_model_baseline([],table,10,20))

    def test_literal_lambda_is_validated_and_reaches_router_unchanged(self):
        from lonn.task_router import rank_models
        self.credential()
        for value in (0.0, 0.2, 1.0):
            self.assertEqual(self.client.put('/api/settings', json=dict(self.account['settings'],lam=value,enforce_lam=True)).status_code, 200)
            with patch('lonn.api.rank_models', wraps=rank_models) as rank, patch('lonn.demo.api.execute', side_effect=self.answer):
                result = self.generate(lam=0.99)
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(result.json()['lam'], value)
                self.assertEqual(rank.call_args.kwargs['lam'], value)
        self.assertEqual(self.client.put('/api/settings',json=dict(self.account['settings'],lam=1.01)).status_code,422)
        self.assertEqual(self.client.put('/api/settings',json=dict(self.account['settings'],lam=-0.01)).status_code,422)

    def test_legacy_lambda_migration_preserves_in_range_values(self):
        store = self.app.state.store
        store.settings(dict(store.settings(), lam=20.0))
        personal = store.for_account('demo-personal')
        personal.settings(dict(personal.settings(),lam=0.75))
        store.migrate_lambda()
        self.assertEqual(store.settings()['lam'],0.2)
        self.assertEqual(personal.settings()['lam'],0.75)

    def test_enterprise_hierarchy_key_access_lambda_precedence_and_attribution(self):
        self.credential()
        project=self.client.post('/api/projects',json={'name':'Support project','parent_id':'team-support'}).json()
        application=self.client.post('/api/applications',json={'name':'chatbot','parent_id':project['id']}).json()
        user='user-engineer'
        memberships={'team_ids':['team-engineering','team-support'],
                     'project_ids':['project-team-engineering',project['id']],
                     'application_ids':['app-team-engineering',application['id']],
                     'team_admin_ids':['team-support'],'application_admin_ids':[application['id']]}
        self.assertEqual(self.client.put(f'/api/users/{user}/memberships',json=memberships).status_code,200)
        other=TestClient(self.app)
        self.assertEqual(other.post('/api/demo/login',json={'email':'robin@nordic.demo','password':'LonnDemo2026!'}).status_code,200)
        personal=other.get('/api/account').json()['api_key']
        personal_header={'Authorization':'Bearer '+personal,'X-Lonn-Application':'chatbot'}
        self.assertEqual(self.client.post('/generate',headers={'Authorization':'Bearer '+personal},json=self.request).status_code,400)
        self.assertEqual(self.client.post('/generate',headers={'Authorization':'Bearer '+personal,'X-Lonn-Application':'content-studio-app'},json=self.request).status_code,403)
        key_header={'Authorization':'Bearer '+application['api_key']}
        self.assertEqual(self.client.post('/generate',headers=dict(key_header,**{'X-Lonn-Application':'engineering-app'}),json=self.request).status_code,403)
        self.assertEqual(other.put('/api/preference',json={'lam':0.35}).status_code,200)
        self.assertEqual(other.get('/api/applications/'+application['id']+'/lambda').json(),{'lam':0.35,'source':'user','locked':False})
        self.assertEqual(other.put('/api/applications/'+application['id']+'/lambda',json={'lam':0.45}).status_code,200)
        self.assertEqual(other.put('/api/teams/team-support/lambda',json={'lam':0.55}).status_code,200)
        with patch('lonn.demo.api.execute',side_effect=self.answer):
            for headers in (personal_header,key_header):
                result=self.client.post('/generate',headers=headers,json=self.request)
                self.assertEqual(result.status_code,200,result.text)
                self.assertEqual(result.json()['lam'],0.55)
                self.assertEqual(result.headers['X-Lonn-Lambda-Source'],'team')
        rows=[r for r in self.client.get('/api/usage').json() if r['source']=='live']
        self.assertEqual(len(rows),2)
        self.assertEqual({r['user_id'] for r in rows},{None,user})
        self.assertTrue(all((r['customer_id'],r['team_id'],r['project_id'],r['application_id'])==('demo-nordic','team-support',project['id'],application['id']) for r in rows))
        self.assertEqual(self.client.put('/api/settings',json=dict(self.account['settings'],lam=0.65,enforce_lam=True)).status_code,200)
        self.assertEqual(other.get('/api/applications/'+application['id']+'/lambda').json(),{'lam':0.65,'source':'organization','locked':True})
        with patch('lonn.demo.api.execute',side_effect=self.answer):
            self.assertEqual(self.client.post('/generate',headers=personal_header,json=self.request).json()['lam'],0.65)
        self.client.put('/api/settings',json=dict(self.account['settings'],enforce_lam=False))
        other.put('/api/teams/team-support/lambda',json={'lam':None})
        self.assertEqual(other.get('/api/applications/'+application['id']+'/lambda').json()['source'],'application')
        other.put('/api/applications/'+application['id']+'/lambda',json={'lam':None})
        self.assertEqual(other.get('/api/applications/'+application['id']+'/lambda').json()['source'],'user')
        other.put('/api/preference',json={'lam':None})
        self.assertEqual(other.get('/api/applications/'+application['id']+'/lambda').json(),{'lam':0.2,'source':'default','locked':False})
        self.assertEqual(other.put('/api/teams/team-engineering/lambda',json={'lam':0.4}).status_code,403)
        self.assertEqual(other.put('/api/applications/app-team-engineering/lambda',json={'lam':0.4}).status_code,403)
        memberships['application_ids'].remove(application['id']);memberships['application_admin_ids']=[]
        self.assertEqual(self.client.put(f'/api/users/{user}/memberships',json=memberships).status_code,200)
        self.assertEqual(self.client.post('/generate',headers=personal_header,json=self.request).status_code,403)
        other.close()

    def test_connection_resolves_key_application_policy_and_access(self):
        usage_before = len(self.app.state.store.history())
        self.assertEqual(self.client.get('/connection').status_code, 401)
        self.assertEqual(self.client.get('/connection', headers={'Authorization': 'Bearer invalid'}).status_code, 401)
        self.assertEqual(self.client.get('/connection', headers={'Authorization': self.headers['Authorization']}).status_code, 400)
        hierarchy = self.client.get('/api/hierarchy').json()
        application = next(a for a in hierarchy['applications'] if a['id'] == 'app-team-engineering')
        app_headers = {'Authorization': 'Bearer ' + application['api_key']}
        self.client.put('/api/preference', json={'lam': 0.35})
        personal = self.client.get('/connection', headers=self.headers)
        self.assertEqual(personal.status_code, 200, personal.text)
        self.assertEqual(personal.json(), {
            'application': {'id': application['id'], 'name': application['name']},
            'key_type': 'personal', 'lam': 0.35, 'source': 'user', 'locked': False})
        app = self.client.get('/connection', headers=app_headers).json()
        self.assertEqual((app['key_type'], app['lam'], app['source'], app['locked']),
                         ('application', 0.2, 'default', False))
        self.assertEqual(self.client.put('/api/projects/project-team-engineering/lambda',
                                        json={'lam': 0.47, 'lock': True}).status_code, 200)
        for headers in (self.headers, app_headers):
            context = self.client.get('/connection', headers=headers).json()
            self.assertEqual((context['lam'], context['source'], context['locked']), (0.47, 'project', True))
        self.assertEqual(self.client.get('/connection', headers=dict(app_headers,
                         **{'X-Lonn-Application': 'content-studio-app'})).status_code, 403)
        self.login_as('robin@nordic.demo')
        key = self.client.get('/api/account').json()['api_key']
        self.assertEqual(self.client.get('/connection', headers={
            'Authorization': 'Bearer ' + key, 'X-Lonn-Application': 'content-studio-app'}).status_code, 403)
        self.assertEqual(len(self.app.state.store.history()), usage_before)

    def test_project_and_application_lambda_configuration_lock_and_inheritance(self):
        self.credential()
        project = 'project-team-engineering'
        application = 'app-team-engineering'
        project_url = f'/api/projects/{project}/lambda'
        app_url = f'/api/applications/{application}/lambda'
        state_url = f'/api/applications/{application}/lambda'
        with patch('lonn.demo.api.execute', side_effect=self.answer):
            # A project-local value is inherited by an unconfigured application.
            self.assertEqual(self.client.put(project_url,json={'lam':0.123456,'lock':False}).status_code,200)
            self.assertEqual(self.client.get(state_url).json(),{'lam':0.123456,'source':'project','locked':False})
            self.assertEqual(self.generate().json()['lam'],0.123456)
            # Unlocked descendants and request preferences can override a local default.
            self.assertEqual(self.client.put(app_url,json={'lam':0.654321,'lock':False}).status_code,200)
            self.assertEqual(self.generate().json()['lam'],0.654321)
            self.assertEqual(self.generate(lam=0.4).json()['lam'],0.4)
            self.assertEqual(self.client.put(app_url,json={'lam':0.654321,'lock':True}).status_code,200)
            self.assertEqual(self.generate(lam=0.4).json()['lam'],0.654321)
            # A project lock wins and prevents editing the lower level.
            self.assertEqual(self.client.put(project_url,json={'lam':0.21,'lock':True}).status_code,200)
            self.assertEqual(self.generate(lam=0.4).json()['lam'],0.21)
            self.assertEqual(self.client.put(app_url,json={'lam':0.1,'lock':False}).status_code,409)
            hierarchy=self.client.get('/api/hierarchy').json()
            app=next(a for a in hierarchy['applications'] if a['id']==application)
            self.assertEqual(app['parent_lock'],'project')
            # Inherit clears the project setting and exposes the existing app lock.
            self.assertEqual(self.client.put(project_url,json={'lam':None,'lock':False}).status_code,200)
            self.assertEqual(self.generate().json()['lam'],0.654321)
            self.assertEqual(self.client.put(app_url,json={'lam':None,'lock':False}).status_code,200)
            self.assertEqual(self.generate().json()['lam'],0.2)
            self.assertEqual(self.client.put(project_url,json={'lam':1.001,'lock':True}).status_code,422)
            self.assertEqual(self.client.put(project_url,json={'lam':0.0,'lock':True}).status_code,200)
            self.assertEqual(self.generate(lam=0.9).json()['lam'],0.0)
            # Team and organization policies lock the controls and win at runtime.
            self.assertEqual(self.client.put('/api/teams/team-engineering/lambda',json={'lam':0.75}).status_code,200)
            self.assertEqual(self.client.put(project_url,json={'lam':0.3,'lock':False}).status_code,409)
            self.assertEqual(self.generate().json()['lam'],0.75)
            self.assertEqual(self.client.put('/api/settings',json=dict(self.account['settings'],lam=0.85,enforce_lam=True)).status_code,200)
            self.assertEqual(self.client.put('/api/teams/team-engineering/lambda',json={'lam':None}).status_code,409)
            self.assertEqual(self.generate().json()['lam'],0.85)
        reopened=Store(self.temp.name)
        from sqlmodel import Session
        from lonn.demo.enterprise import Project
        with Session(reopened.engine) as session:
            saved=session.get(Project,project)
            self.assertEqual((saved.configured_lam,saved.lam_locked),(0.0,True))
        reopened.engine.dispose()

    def test_private_user_can_manage_own_project_and_application_policy(self):
        self.login_as('sam@personal.demo')
        self.assertEqual(self.client.put('/api/projects/project-team-personal/lambda',json={'lam':0.31,'lock':False}).status_code,200)
        self.assertEqual(self.client.put('/api/applications/app-team-personal/lambda',json={'lam':0.42,'lock':True}).status_code,200)
        self.assertEqual(self.client.get('/api/applications/app-team-personal/lambda').json(),{'lam':0.42,'source':'application','locked':True})
        self.assertEqual(self.client.put('/api/projects/project-team-engineering/lambda',json={'lam':0.1,'lock':True}).status_code,403)

    def test_guide_sdk_pool_and_eligible_models_remain_distinct(self):
        with LonnRouter(base_url='http://testserver', api_key=self.account['api_key'],
                        application='engineering-app', http_client=self.client) as sdk:
            configured = sdk.model_pool()
            self.assertTrue(configured)
            self.assertEqual({m.id for m in configured}, {m.id for m in sdk.models()})
            settings = dict(self.account['settings'], require_zdr=True)
            self.assertEqual(self.client.put('/api/settings', json=settings).status_code, 200)
            self.assertEqual({m.id for m in configured}, {m.id for m in sdk.model_pool()})
            self.assertEqual(sdk.models(), [])
            disabled = configured[0].id
            self.client.put('/api/settings', json=dict(settings, disabled_models=[disabled]))
            self.assertNotIn(disabled, [m.id for m in sdk.model_pool()])

    def test_guide_direct_model_never_bypasses_policy_or_falls_back(self):
        self.credential()
        available = self.client.get('/models', headers=self.headers).json()
        chosen = available[-1]['id']
        with patch('lonn.demo.api.execute', side_effect=self.answer) as execute:
            with LonnRouter(base_url='http://testserver', api_key=self.account['api_key'],
                            application='engineering-app', http_client=self.client) as sdk:
                result = sdk.generate('Solve 2 + 2', model=chosen, signal=self.request['signal'])
            self.assertEqual(result.model.id, chosen)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(execute.call_args.args[1].id, chosen)
        row = next(r for r in self.client.get('/api/usage').json() if r['source']=='live')
        self.assertEqual(row['details']['selection'], 'direct')
        team = next(t for t in self.client.get('/api/teams').json() if t['id']=='team-engineering')
        self.client.put('/api/teams/team-engineering', json={'name':team['name'], 'enabled':True, 'disabled_models':[chosen]})
        with patch('lonn.demo.api.execute') as execute:
            self.assertEqual(self.generate(model=chosen).status_code, 403)
            execute.assert_not_called()
        self.client.put('/api/teams/team-engineering', json={'name':team['name'], 'enabled':True, 'disabled_models':[]})
        with patch('lonn.demo.api.execute', side_effect=RuntimeError('provider unavailable')) as execute:
            self.assertEqual(self.generate(model=chosen).status_code, 502)
            self.assertEqual(execute.call_count, 1)
        self.client.put('/api/settings', json=dict(self.account['settings'], disabled_models=[chosen]))
        with patch('lonn.demo.api.execute') as execute:
            self.assertEqual(self.generate(model=chosen).status_code, 403)
            execute.assert_not_called()
        self.client.put('/api/settings', json=dict(self.account['settings'], require_zdr=True))
        with patch('lonn.demo.api.execute') as execute:
            self.assertEqual(self.generate(model=chosen).status_code, 422)
            execute.assert_not_called()

    def test_guide_lambda_request_and_administrator_lock(self):
        self.credential()
        with patch('lonn.demo.api.execute', side_effect=self.answer):
            with LonnRouter(base_url='http://testserver', api_key=self.account['api_key'],
                            application='engineering-app', http_client=self.client) as sdk:
                result = sdk.generate('Solve 2 + 2', lam=0.43, signal=self.request['signal'])
                self.assertEqual(result.lam, 0.43)
                self.client.put('/api/settings', json=dict(self.account['settings'], lam=0.8, enforce_lam=True))
                self.assertEqual(sdk.generate('Solve 2 + 2', lam=0.43, signal=self.request['signal']).lam, 0.8)
                self.assertEqual(self.generate(lam=1.01).status_code, 422)
        rows = [r for r in self.client.get('/api/usage').json() if r['source']=='live']
        self.assertEqual({r['details']['lambda_source'] for r in rows}, {'request','organization'})
        self.client.put('/api/settings', json=dict(self.account['settings'], enforce_lam=False))
        self.assertEqual(self.generate(lam=1.01).status_code, 422)

    def test_async_sdk_auth(self):
        self.credential()
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app)) as transport:
                async with AsyncLonnRouter(base_url='http://testserver', api_key=self.account['api_key'], application='engineering-app', http_client=transport) as sdk:
                    return await sdk.generate('2+2', signal=self.request['signal'], verify=False)
        with patch('lonn.demo.api.execute', side_effect=self.answer):
            self.assertEqual(asyncio.run(run()).content, '4')


if __name__ == '__main__':
    unittest.main()
