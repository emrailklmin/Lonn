"""Authenticated local demo API around Prototype 3.0's unchanged routing pipeline."""
import json
from dataclasses import replace
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter, time
from uuid import uuid4

from fastapi import Depends, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr
from starlette.middleware.trustedhost import TrustedHostMiddleware

from lonn.api import GenerateRequest, GenerateResponse, RouteRequest, RouteResponse, create_app as core_app
from lonn.executor import execute
from .policy import Settings, catalog_status
from .pricing import BENCHMARK_COSTS, request_billing, token_cost
from .baseline import top_model_baseline
from .auth import verify_password
from .store import ROOT, Store, User, Team
from .enterprise import (Project, Application, TeamMember, ProjectMember, ApplicationMember,
                         application_context, has_access, lambda_state, new_id, new_key)
from sqlmodel import Session, select


def create_app(directory=None):
    app = core_app()
    core_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with core_lifespan(app):
            app.state.store = Store(directory)
            app.state.sessions = {}
            yield
            app.state.store.engine.dispose()

    app.router.lifespan_context = lifespan
    app.title = 'Lönn customer demo'
    # Never expose the original unauthenticated generation routes on the demo server.
    app.router.routes = [r for r in app.router.routes if r.path not in ('/route', '/generate', '/models')]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', 'testserver'])

    @app.middleware('http')
    async def protect_local_management(request: Request, call_next):
        origin = request.headers.get('origin')
        if request.method not in ('GET', 'HEAD', 'OPTIONS') and origin and origin not in (
            'http://127.0.0.1:8000', 'http://localhost:8000',
            'http://127.0.0.1:5173', 'http://localhost:5173'):
            return Response('Origin not allowed', status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        if request.url.path in ('/api/provider', '/api/demo/login', '/api/users'):
            return JSONResponse(status_code=422, content={'detail': 'Invalid input'})
        return await request_validation_exception_handler(request, exc)

    def viewer(request: Request):
        session = app.state.sessions.get(request.cookies.get('lonn_session'))
        if not session or session[1] < time():
            raise HTTPException(401, 'Sign in to your demo account')
        user = app.state.store.user(user_id=session[0])
        if user is None or not user.enabled:
            raise HTTPException(401, 'User access is disabled')
        return app.state.store.for_account(user.customer_id), user

    def management(identity=Depends(viewer)):
        store, user = identity
        if user.role not in ('enterprise_admin', 'private_user'):
            raise HTTPException(403, 'Your administrator manages these settings')
        return store

    def administrator(identity=Depends(viewer)):
        store, user = identity
        if user.role != 'enterprise_admin':
            raise HTTPException(403, 'Enterprise administrator access required')
        return store

    def user_team(store, user):
        return next((t for t in store.teams() if t.id == user.team_id), None)

    def customer(authorization: str = Header(default=''), x_lonn_application: str | None = Header(default=None)):
        root = app.state.store
        for account in root.accounts():
            store = root.for_account(account.id)
            for user in store.users():
                if user.personal_key and secrets.compare_digest(authorization, 'Bearer ' + user.personal_key):
                    if not user.enabled:
                        raise HTTPException(401, 'Invalid or revoked Lönn API key')
                    if not x_lonn_application:
                        raise HTTPException(400, 'Personal keys require an application')
                    context = application_context(store, x_lonn_application)
                    if not context or not has_access(store, user, *context):
                        raise HTTPException(403, 'Application access denied')
                    application, project, team = context
                    if not application.enabled or not team.enabled:
                        raise HTTPException(403, 'Application or team access disabled')
                    return store, team, project, application, user
            with Session(store.engine) as session:
                apps = session.exec(select(Application).where(Application.customer_id == account.id)).all()
            for application in apps:
                if secrets.compare_digest(authorization, 'Bearer ' + application.api_key):
                    if x_lonn_application and x_lonn_application not in (application.id, application.name):
                        raise HTTPException(403, 'Application keys cannot select another application')
                    context = application_context(store, application.id)
                    if not context or not application.enabled or not context[2].enabled:
                        raise HTTPException(401, 'Invalid or revoked Lönn API key')
                    application, project, team = context
                    return store, team, project, application, None
        raise HTTPException(401, 'Invalid or revoked Lönn API key')

    def effective_settings(store, team):
        settings = store.settings()
        if team:
            settings['disabled_models'] = sorted(set(settings['disabled_models']) | set(json.loads(team.disabled_models)))
        return settings

    def pool(store, settings):
        ids = {m['id'] for m in catalog(store, settings) if m['eligible']}
        models = [m for m in app.state.models if m.id in ids]
        if not models:
            raise HTTPException(422, 'No eligible models. Review account privacy and model pool settings.')
        return models

    def catalog(store, settings):
        return [catalog_status(e, settings, app.state.table) for e in store.catalog()]

    class Login(BaseModel):
        email: str = Field(max_length=200)
        password: SecretStr = Field(max_length=200)

    @app.post('/api/demo/login')
    def login(value: Login, response: Response):
        user = app.state.store.user(email=value.email.strip().lower())
        if user is None or not user.enabled or not verify_password(value.password.get_secret_value(), user.password_hash):
            raise HTTPException(401, 'Invalid email or password')
        token = secrets.token_urlsafe(32)
        app.state.sessions[token] = (user.id, time() + 86400)
        response.set_cookie('lonn_session', token, httponly=True, samesite='strict', max_age=86400)
        return {'name': user.name, 'role': user.role}

    @app.post('/api/demo/logout')
    def logout(request: Request, response: Response):
        app.state.sessions.pop(request.cookies.get('lonn_session'), None)
        response.delete_cookie('lonn_session')
        return {'ok': True}

    @app.get('/api/account')
    def account(identity=Depends(viewer)):
        store, user = identity
        a = store.account()
        team = user_team(store, user)
        key = user.personal_key
        return {'id': a.id, 'name': a.name, 'api_key': key, 'settings': effective_settings(store, team),
                'user': {'id': user.id, 'name': user.name, 'email': user.email, 'role': user.role,
                         'team_id': user.team_id, 'team_name': team.name if team else None,
                         'lam_preference': user.lam_preference},
                'provider_configured': bool(store.credential()), 'endpoint': 'http://127.0.0.1:8000'}

    @app.put('/api/settings')
    def settings(value: Settings, store=Depends(management)):
        known = {m['id'] for m in store.catalog()}
        if set(value.disabled_models) - known:
            raise HTTPException(422, 'Unknown model ID')
        return store.settings(value.model_dump())

    class Credential(BaseModel):
        key: SecretStr = Field(min_length=1, max_length=512)

    @app.put('/api/provider')
    def provider(value: Credential, store=Depends(administrator)):
        key = value.key.get_secret_value().strip()
        if not key or '\n' in key or '\r' in key:
            raise HTTPException(422, 'Enter a valid provider key')
        store.save_credential(key)
        return {'configured': True}

    @app.delete('/api/provider')
    def clear_provider(store=Depends(administrator)):
        store.save_credential('')
        return {'configured': False}

    @app.post('/api/key/rotate')
    def rotate(identity=Depends(viewer)):
        store, user = identity
        with Session(store.engine) as session:
            user = session.get(User, user.id)
            user.personal_key = new_key('personal')
            session.add(user)
            session.commit()
            return {'api_key': user.personal_key}

    @app.get('/api/catalog')
    def model_catalog(identity=Depends(viewer)):
        store, user = identity
        return catalog(store, effective_settings(store, user_team(store, user)))

    @app.get('/api/usage')
    def usage(identity=Depends(viewer)):
        store, user = identity
        rows = store.history()
        if user.role != 'enterprise_user':
            return rows
        with Session(store.engine) as session:
            app_ids = {m.application_id for m in session.exec(select(ApplicationMember).where(ApplicationMember.user_id == user.id)).all()}
        return [r for r in rows if r['application_id'] in app_ids]

    class TeamCreate(BaseModel):
        name: str = Field(min_length=1, max_length=80)

    class TeamUpdate(TeamCreate):
        enabled: bool
        disabled_models: list[str]

    def team_view(team, store):
        value = team.model_dump(exclude={'api_key'})
        value['disabled_models'] = json.loads(team.disabled_models)
        value['eligible_model_ids'] = [m['id'] for m in catalog(store, effective_settings(store, team)) if m['eligible']] if team.enabled else []
        return value

    @app.get('/api/teams')
    def teams(identity=Depends(viewer)):
        store, user = identity
        with Session(store.engine) as session:
            memberships = {m.team_id for m in session.exec(select(TeamMember).where(TeamMember.user_id == user.id)).all()}
        return [team_view(t, store) for t in store.teams() if user.role in ('enterprise_admin','private_user') or t.id in memberships]

    @app.post('/api/teams', status_code=201)
    def create_team(value: TeamCreate, store=Depends(administrator)):
        if not value.name.strip():
            raise HTTPException(422, 'Team name must not be blank')
        team = store.create_team(value.name.strip())
        store.seed_hierarchy()
        return team_view(team, store)

    @app.put('/api/teams/{team_id}')
    def update_team(team_id: str, value: TeamUpdate, store=Depends(administrator)):
        if not value.name.strip() or set(value.disabled_models) - {m['id'] for m in store.catalog()}:
            raise HTTPException(422, 'Invalid team name or model selection')
        team = store.update_team(team_id, **dict(value.model_dump(), name=value.name.strip()))
        if team is None:
            raise HTTPException(404, 'Team not found')
        return team_view(team, store)

    @app.post('/api/teams/{team_id}/rotate')
    def rotate_team(team_id: str, store=Depends(administrator)):
        raise HTTPException(410, 'Rotate the application key instead')

    def public_user(user):
        return user.model_dump(exclude={'password_hash', 'personal_key'})

    class UserCreate(BaseModel):
        name: str = Field(min_length=1, max_length=80)
        email: str = Field(min_length=3, max_length=200)
        password: SecretStr = Field(min_length=10, max_length=200)
        team_id: str

    class UserUpdate(BaseModel):
        team_id: str
        enabled: bool

    @app.get('/api/users')
    def users(store=Depends(administrator)):
        return [public_user(u) for u in store.users()]

    @app.post('/api/users', status_code=201)
    def create_user(value: UserCreate, store=Depends(administrator)):
        if value.team_id not in {t.id for t in store.teams()} or not value.name.strip() or '@' not in value.email:
            raise HTTPException(422, 'Valid name, email and organization team required')
        email = value.email.strip().lower()
        if store.user(email=email):
            raise HTTPException(409, 'Email already exists')
        user = store.create_user(name=value.name.strip(), email=email,
                           password=value.password.get_secret_value(), team_id=value.team_id)
        store.seed_hierarchy()
        return public_user(store.user(user_id=user.id))

    @app.put('/api/users/{user_id}')
    def update_user(user_id: str, value: UserUpdate, store=Depends(administrator)):
        if value.team_id not in {t.id for t in store.teams()}:
            raise HTTPException(422, 'Unknown organization team')
        user = store.update_user(user_id, **value.model_dump())
        if user is None:
            raise HTTPException(404, 'Editable enterprise user not found')
        return public_user(user)

    class LambdaPolicy(BaseModel):
        lam: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False, strict=True)

    class NamedParent(BaseModel):
        name: str = Field(min_length=1, max_length=80)
        parent_id: str

    class ScopedLambdaPolicy(LambdaPolicy):
        lock: bool = True  # Existing API calls with only lam retain enforcement.

    class Memberships(BaseModel):
        team_ids: list[str]
        project_ids: list[str]
        application_ids: list[str]
        team_admin_ids: list[str] = []
        application_admin_ids: list[str] = []

    @app.get('/api/hierarchy')
    def hierarchy(identity=Depends(viewer)):
        store, user = identity
        with Session(store.engine) as session:
            projects = session.exec(select(Project).where(Project.customer_id == store.customer_id)).all()
            apps = session.exec(select(Application).where(Application.customer_id == store.customer_id)).all()
            team_ids = {t.id for t in store.teams()}
            project_ids = {p.id for p in projects}
            application_ids = {a.id for a in apps}
            tm = [m for m in session.exec(select(TeamMember)).all() if m.team_id in team_ids]
            pm = [m for m in session.exec(select(ProjectMember)).all() if m.project_id in project_ids]
            am = [m for m in session.exec(select(ApplicationMember)).all() if m.application_id in application_ids]
            if user.role == 'enterprise_user':
                apps = [a for a in apps if any(m.user_id == user.id and m.application_id == a.id for m in am)]
                projects = [p for p in projects if any(m.user_id == user.id and m.project_id == p.id for m in pm)]
            teams_by_id = {t.id: t for t in store.teams()}
            projects_by_id = {p.id: p for p in projects}
            organization = store.settings()
            def project_view(p):
                team = teams_by_id[p.team_id]
                parent = lambda_state(store, team)
                return dict(p.model_dump(), lambda_state=lambda_state(store, team, p),
                            parent_lock=parent['source'] if parent['locked'] else None)
            def app_view(a):
                p = projects_by_id[a.project_id]
                team = teams_by_id[p.team_id]
                parent = lambda_state(store, team, p)
                return dict(a.model_dump(exclude={'api_key'}),
                            lambda_state=lambda_state(store, team, p, a),
                            parent_lock=parent['source'] if parent['locked'] else None,
                            **({'api_key': a.api_key} if user.role == 'enterprise_admin' else {}))
            return {'organization_lock': bool(organization.get('enforce_lam', False)),
                    'projects': [project_view(p) for p in projects],
                    'applications': [app_view(a) for a in apps],
                    'team_memberships': [m.model_dump() for m in tm if user.role == 'enterprise_admin' or m.user_id == user.id],
                    'project_memberships': [m.model_dump() for m in pm if user.role == 'enterprise_admin' or m.user_id == user.id],
                    'application_memberships': [m.model_dump() for m in am if user.role == 'enterprise_admin' or m.user_id == user.id]}

    @app.post('/api/projects', status_code=201)
    def create_project(value: NamedParent, store=Depends(administrator)):
        with Session(store.engine) as session:
            team = session.get(Team, value.parent_id)
            if not team or team.customer_id != store.customer_id or not value.name.strip():
                raise HTTPException(422, 'Valid team and project name required')
            project = Project(id=new_id(), customer_id=store.customer_id, team_id=team.id, name=value.name.strip())
            session.add(project); session.commit(); session.refresh(project)
            return project

    @app.post('/api/applications', status_code=201)
    def create_application(value: NamedParent, store=Depends(administrator)):
        with Session(store.engine) as session:
            project = session.get(Project, value.parent_id)
            if not project or project.customer_id != store.customer_id or not value.name.strip():
                raise HTTPException(422, 'Valid project and application name required')
            if session.exec(select(Application).where(Application.customer_id == store.customer_id, Application.name == value.name.strip())).first():
                raise HTTPException(409, 'Application name already exists')
            application = Application(id=new_id(), customer_id=store.customer_id, project_id=project.id,
                                      name=value.name.strip(), api_key=new_key('app'))
            session.add(application); session.commit(); session.refresh(application)
            return application

    @app.post('/api/applications/{application_id}/rotate')
    def rotate_application(application_id: str, store=Depends(administrator)):
        with Session(store.engine) as session:
            application = session.get(Application, application_id)
            if not application or application.customer_id != store.customer_id:
                raise HTTPException(404, 'Application not found')
            application.api_key = new_key('app'); session.add(application); session.commit()
            return {'api_key': application.api_key}

    @app.put('/api/users/{user_id}/memberships')
    def set_memberships(user_id: str, value: Memberships, store=Depends(administrator)):
        with Session(store.engine) as session:
            target = session.get(User, user_id)
            if not target or target.customer_id != store.customer_id or target.role != 'enterprise_user':
                raise HTTPException(404, 'Enterprise user not found')
            teams = {t.id: t for t in store.teams()}
            projects = {p.id:p for p in session.exec(select(Project).where(Project.customer_id == store.customer_id)).all()}
            apps = {a.id:a for a in session.exec(select(Application).where(Application.customer_id == store.customer_id)).all()}
            if (set(value.team_ids)-teams.keys() or set(value.project_ids)-projects.keys() or
                set(value.application_ids)-apps.keys() or set(value.team_admin_ids)-set(value.team_ids) or
                set(value.application_admin_ids)-set(value.application_ids) or
                any(projects[p].team_id not in value.team_ids for p in value.project_ids) or
                any(apps[a].project_id not in value.project_ids for a in value.application_ids)):
                raise HTTPException(422, 'Memberships must belong to this organization and include their parent scopes')
            for model in (TeamMember, ProjectMember, ApplicationMember):
                for old in session.exec(select(model).where(model.user_id == user_id)).all(): session.delete(old)
            session.flush()
            target.team_id = value.team_ids[0] if value.team_ids else None
            session.add(target)
            for tid in set(value.team_ids): session.add(TeamMember(user_id=user_id, team_id=tid, role='team_admin' if tid in value.team_admin_ids else 'member'))
            for pid in set(value.project_ids): session.add(ProjectMember(user_id=user_id, project_id=pid))
            for aid in set(value.application_ids): session.add(ApplicationMember(user_id=user_id, application_id=aid, role='application_admin' if aid in value.application_admin_ids else 'member'))
            session.commit()
        return {'ok': True}

    @app.get('/api/preference')
    def preference(identity=Depends(viewer)):
        return {'lam': identity[1].lam_preference}

    @app.put('/api/preference')
    def set_preference(value: LambdaPolicy, identity=Depends(viewer)):
        store, user = identity
        with Session(store.engine) as session:
            user = session.get(User, user.id); user.lam_preference = value.lam
            session.add(user); session.commit()
        return {'lam': value.lam}

    @app.get('/api/applications/{application_id}/lambda')
    def application_lambda(application_id: str, identity=Depends(viewer)):
        store, user = identity
        context = application_context(store, application_id)
        if not context or not has_access(store, user, *context):
            raise HTTPException(403, 'Application access denied')
        application, project, team = context
        return lambda_state(store, team, project, application, user)

    @app.put('/api/teams/{team_id}/lambda')
    def team_lambda(team_id: str, value: LambdaPolicy, identity=Depends(viewer)):
        store, user = identity
        with Session(store.engine) as session:
            team = session.get(Team, team_id)
            membership = session.get(TeamMember, (user.id, team_id))
            if not team or team.customer_id != store.customer_id or (user.role != 'enterprise_admin' and (not membership or membership.role != 'team_admin')):
                raise HTTPException(403, 'Team administrator access required')
            if store.settings().get('enforce_lam', False):
                raise HTTPException(409, 'λ is enforced by organization')
            team.enforced_lam = value.lam; session.add(team); session.commit()
        return {'lam': value.lam}

    @app.put('/api/projects/{project_id}/lambda')
    def project_lambda(project_id: str, value: ScopedLambdaPolicy, identity=Depends(viewer)):
        store, user = identity
        with Session(store.engine) as session:
            project = session.get(Project, project_id)
            team = session.get(Team, project.team_id) if project else None
            member = session.get(TeamMember, (user.id, team.id)) if team else None
            if not project or project.customer_id != store.customer_id or not team or (user.role not in ('enterprise_admin', 'private_user') and (not member or member.role != 'team_admin')):
                raise HTTPException(403, 'Project access requires an organization or team administrator')
            parent = lambda_state(store, team)
            if parent['locked']:
                raise HTTPException(409, f"λ is enforced by {parent['source']}")
            project.configured_lam = value.lam
            project.lam_locked = value.lock if value.lam is not None else False
            session.add(project); session.commit()
        return {'lam': value.lam, 'locked': value.lock if value.lam is not None else False}

    @app.put('/api/applications/{application_id}/lambda')
    def app_lambda(application_id: str, value: ScopedLambdaPolicy, identity=Depends(viewer)):
        store, user = identity
        with Session(store.engine) as session:
            application = session.get(Application, application_id)
            member = session.get(ApplicationMember, (user.id, application_id))
            if not application or application.customer_id != store.customer_id or (user.role not in ('enterprise_admin', 'private_user') and (not member or member.role != 'application_admin')):
                raise HTTPException(403, 'Application administrator access required')
            project = session.get(Project, application.project_id)
            team = session.get(Team, project.team_id) if project else None
            if not team or team.customer_id != store.customer_id:
                raise HTTPException(403, 'Application access denied')
            parent = lambda_state(store, team, project)
            if parent['locked']:
                raise HTTPException(409, f"λ is enforced by {parent['source']}")
            application.enforced_lam = value.lam
            application.lam_locked = value.lock if value.lam is not None else False
            session.add(application); session.commit()
        return {'lam': value.lam, 'locked': value.lock if value.lam is not None else False}

    def request_lambda(request, store, team, project, application, user):
        state = lambda_state(store, team, project, application, user)
        if 'lam' in request.model_fields_set:
            if request.lam > 1:
                raise HTTPException(422, 'λ must be between 0 and 1 in the demo')
            if not state['locked']:
                return request.lam, 'request'
        return state['lam'], state['source']

    class DemoGenerateRequest(GenerateRequest):
        model: str | None = Field(default=None, min_length=1, max_length=300)

    @app.get('/connection')
    def connection(identity=Depends(customer)):
        store, team, project, application, user = identity
        return {'application': {'id': application.id, 'name': application.name},
                'key_type': 'personal' if user else 'application',
                **lambda_state(store, team, project, application, user)}

    @app.get('/models/pool')
    def configured_pool(identity=Depends(customer)):
        store, team, project, application, user = identity
        settings = effective_settings(store, team)
        # Pool membership includes organization/team selections; privacy rules
        # are applied separately to form the eligible set.
        settings.update(require_zdr=False, forbid_training=False, region='any', max_retention_days=None)
        ids = {m['id'] for m in catalog(store, settings) if m['eligible']}
        return [m for m in app.state.models if m.id in ids]

    @app.get('/models')
    def models(identity=Depends(customer)):
        store, team, project, application, user = identity
        settings = effective_settings(store, team)
        ids = {m['id'] for m in catalog(store, settings) if m['eligible']}
        return [m for m in app.state.models if m.id in ids]

    @app.post('/route', response_model=RouteResponse)
    def route(request: RouteRequest, identity=Depends(customer)):
        store, team, project, application, user = identity
        settings = effective_settings(store, team)
        settings['lam'], _ = request_lambda(request, store, team, project, application, user)
        return app.state.route_request(request.model_copy(update={'lam': settings['lam']}), pool(store, settings))

    class DemoGenerateResponse(GenerateResponse):
        billing: dict

    @app.post('/generate', response_model=DemoGenerateResponse)
    def generate(request: DemoGenerateRequest, response: Response, identity=Depends(customer)):
        store, team, project, application, user = identity
        started = perf_counter()
        request_id = str(uuid4())
        settings = effective_settings(store, team)
        settings['lam'], lambda_source = request_lambda(request, store, team, project, application, user)
        calls = []
        attempted = []
        result = None
        failure = None
        baseline = None
        billing = None
        model_costs = []
        def finalize_billing():
            nonlocal calls, billing
            billing = request_billing(model_costs, baseline['model_id'] if baseline else None)
            # Allocate the request fee across completed attempts so their costs still sum to the total.
            remaining = billing['service_fee_usd']
            for index, value in enumerate(calls):
                cost = float(model_costs[index][1])
                fee = (remaining if index == len(calls) - 1 else
                       billing['service_fee_usd'] * cost / billing['model_cost_usd'] if billing['model_cost_usd'] else 0.0)
                remaining -= fee
                calls[index] = replace(value, usage=replace(value.usage, total_cost=cost + fee, service_fee=fee))
            return billing

        def executor(prompt, model, **kwargs):
            attempted.append(model.id)
            value = execute(prompt, model, api_key=credential, **kwargs)
            cost = token_cost(value.usage.prompt_tokens, value.usage.completion_tokens,
                              model.cost_per_m_input, model.cost_per_m_output)
            model_costs.append((model.id, cost))
            value = replace(value, usage=replace(value.usage,
                input_cost=float(token_cost(value.usage.prompt_tokens, 0, model.cost_per_m_input, 0)),
                output_cost=float(token_cost(0, value.usage.completion_tokens, 0, model.cost_per_m_output)),
                total_cost=float(cost), service_fee=0.0, cost_basis='calculated'))
            calls.append(value)
            return value
        try:
            eligible = pool(store, settings)
            baseline = top_model_baseline([m for m in eligible if request.max_tokens < m.context_window],
                                          app.state.table, 0, 0)
            selected_pool = eligible
            if request.model is not None:
                selected_pool = [m for m in eligible if m.id == request.model]
                if not selected_pool:
                    raise HTTPException(403, 'Requested model is not eligible for this application')
            credential = store.credential()
            if not credential:
                raise HTTPException(503, 'Configure the presenter OpenRouter credential in Settings')
            result = app.state.generate_request(request.model_copy(update={'lam': settings['lam']}),
                                                selected_pool, executor)
            finalize_billing()
            for attempt, call in zip((a for a in result.attempts if a.check != 'error'), calls):
                attempt.cost_usd = call.usage.total_cost
            result = DemoGenerateResponse(**result.model_dump(exclude={'response', 'total_cost_usd'}),
                response=calls[-1], total_cost_usd=billing['total_cost_usd'], billing=billing)
            response.headers['X-Lonn-Request-ID'] = request_id
            response.headers['X-Lonn-Lambda-Source'] = lambda_source
            return result
        except HTTPException as exc:
            failure = exc.status_code
            raise
        except Exception:
            failure = 500
            raise HTTPException(500, 'Request failed; inspect local configuration') from None
        finally:
            if billing is None:
                finalize_billing()
            # No prompt, output, raw provider errors or credentials stored in telemetry.
            detail = {'settings': settings, 'lambda_source': lambda_source, 'selection': 'direct' if request.model else 'routed', 'team_id': team.id, 'attempted_models': attempted,
                      'cost_basis': 'calculated',
                      'billing': billing,
                      'attempt_costs': [dict(model=c.model.id, cost=c.usage.total_cost, basis=c.usage.cost_basis,
                                             input_tokens=c.usage.prompt_tokens, output_tokens=c.usage.completion_tokens,
                                             input_rate_per_m=c.model.cost_per_m_input, output_rate_per_m=c.model.cost_per_m_output,
                                             model_cost_usd=float(model_costs[index][1]),
                                             benchmark_cost_per_690=float(BENCHMARK_COSTS[c.model.id]) if c.model.id in BENCHMARK_COSTS else None,
                                             service_fee=c.usage.service_fee, upstream_provider_cost=c.usage.provider_cost) for index, c in enumerate(calls)],
                      'usage_complete': len(calls) == len(attempted),
                      'http_status': failure or 200}
            if baseline:
                baseline.update(input_tokens=sum(c.usage.prompt_tokens for c in calls),
                                output_tokens=sum(c.usage.completion_tokens for c in calls),
                                estimated_cost_usd=billing['estimated_top_model_cost_usd'])
                detail['baseline'] = baseline
            if result:
                detail.update(signal=result.signal.__dict__, estimates=[e.model_dump() for e in result.estimates],
                              attempts=[a.model_dump() for a in result.attempts], escalated=result.escalated,
                              finish_reason=result.response.finish_reason)
            store.record(id=request_id, team_id=team.id, project_id=project.id,
                application_id=application.id, user_id=user.id if user else None, timestamp=datetime.now(timezone.utc).isoformat(), source='live',
                model=result.model.id if result else (attempted[-1] if attempted else None),
                status='success' if result else 'failed',
                input_tokens=sum(c.usage.prompt_tokens for c in calls),
                output_tokens=sum(c.usage.completion_tokens for c in calls),
                cost_usd=billing['total_cost_usd'],
                latency_ms=(perf_counter() - started) * 1000, details=json.dumps(detail))

    # Serve the built React portal and the visually separate customer application.
    dist = ROOT / 'frontend/dist'
    if dist.exists():
        app.mount('/assets', StaticFiles(directory=dist / 'assets'), name='assets')
        @app.get('/')
        @app.get('/customer')
        def frontend():
            return FileResponse(dist / 'index.html')
    return app


app = create_app()
