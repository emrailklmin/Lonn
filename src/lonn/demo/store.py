"""Account-scoped local persistence. Runtime state never belongs in Git."""
import json
import os
import secrets
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import inspect
from pathlib import Path

import yaml
from sqlmodel import Field, Session, SQLModel, create_engine, select

from lonn.model_pool import DEFAULT_POOL
from .seed import history
from .policy import DEMO_SECURITY, MAX_DEMO_LAMBDA, Settings, normalize_settings
from .pricing import FEE_BASIS
from .auth import password_hash
from .enterprise import Project, Application, TeamMember, ProjectMember, ApplicationMember, new_id, new_key

ROOT = Path(__file__).resolve().parents[3]


class Account(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    api_key: str
    settings: str


class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    customer_id: str = Field(index=True)
    name: str
    email: str = Field(index=True, unique=True)
    role: str
    team_id: str | None = None
    password_hash: str
    enabled: bool = True
    personal_key: str | None = None
    lam_preference: float | None = None


class Team(SQLModel, table=True):
    id: str = Field(primary_key=True)
    customer_id: str = Field(index=True)
    name: str
    api_key: str
    enabled: bool = True
    disabled_models: str = '[]'
    enforced_lam: float | None = None


class Catalog(SQLModel, table=True):
    id: str = Field(primary_key=True)
    metadata_json: str


class Usage(SQLModel, table=True):
    id: str = Field(primary_key=True)
    customer_id: str = Field(index=True)
    team_id: str | None = None
    project_id: str | None = None
    application_id: str | None = None
    user_id: str | None = None
    timestamp: str
    source: str
    model: str | None = None
    status: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0
    latency_ms: float = 0
    details: str = '{}'


class Store:
    def __init__(self, directory=None):
        self.customer_id = "demo-nordic"
        self.directory = Path(directory or os.environ.get('LONN_DEMO_DIR', ROOT / '.local/demo'))
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.engine = create_engine(f'sqlite:///{self.directory / "demo.sqlite3"}',
                                   connect_args={'check_same_thread': False, 'timeout': 30})
        self.migrate_lambda()
        SQLModel.metadata.create_all(self.engine)
        # Only additive schema updates: existing settings, credentials and usage survive.
        additions = {'usage': {'team_id': 'VARCHAR', 'project_id': 'VARCHAR',
                               'application_id': 'VARCHAR', 'user_id': 'VARCHAR'},
                     'user': {'personal_key': 'VARCHAR', 'lam_preference': 'FLOAT'},
                     'team': {'enforced_lam': 'FLOAT'},
                     'project': {'configured_lam': 'FLOAT', 'lam_locked': 'BOOLEAN DEFAULT 0'},
                     'application': {'lam_locked': 'BOOLEAN DEFAULT 1',
                                     'security_overrides': "VARCHAR DEFAULT '{}'"}}
        for table, columns in additions.items():
            existing = {c['name'] for c in inspect(self.engine).get_columns(table)}
            with self.engine.begin() as connection:
                for column, kind in columns.items():
                    if column not in existing:
                        connection.exec_driver_sql(f'ALTER TABLE "{table}" ADD COLUMN {column} {kind}')
        os.chmod(self.directory / 'demo.sqlite3', 0o600)
        self.seed()
        self.seed_teams()
        self.seed_users()
        self.seed_hierarchy()
        self.migrate_seeded_tariff()

    def seed(self):
        seed = json.loads((ROOT / 'data/demo/seed.json').read_text())
        with Session(self.engine) as s:
            if s.get(Account, seed['id']):
                return
            s.add(Account(id=seed['id'], name=seed['name'], api_key='lonn_demo_' + secrets.token_urlsafe(32),
                          settings=Settings().model_dump_json()))
            entries = [entry for entry in yaml.safe_load(DEFAULT_POOL.read_text())
                       if entry['active'] and entry['id'] in DEMO_SECURITY]
            for entry in entries:
                s.add(Catalog(id=entry['id'], metadata_json=json.dumps(entry)))
            for values in history(seed, entries):
                s.add(Usage(**values))
            s.commit()

    def seed_hierarchy(self):
        with Session(self.engine) as s:
            # Migrate legacy keys to the corresponding personal or default-app key.
            migrated_users = set()
            for user in s.exec(select(User)).all():
                if not user.personal_key:
                    migrated_users.add(user.id)
                    account = s.get(Account, user.customer_id)
                    user.personal_key = account.api_key if user.role in ('enterprise_admin', 'private_user') else new_key('personal')
                    s.add(user)
            # A private account also gets one simple workspace for SDK attribution.
            personal = s.get(Account, 'demo-personal')
            if personal and not s.get(Team, 'team-personal'):
                s.add(Team(id='team-personal', customer_id=personal.id, name='Personal', api_key=new_key('app')))
            s.flush()
            for team in s.exec(select(Team)).all():
                project_id, app_id = 'project-' + team.id, 'app-' + team.id
                if not s.get(Project, project_id):
                    s.add(Project(id=project_id, customer_id=team.customer_id, team_id=team.id, name='Default project'))
                if not s.get(Application, app_id):
                    s.add(Application(id=app_id, customer_id=team.customer_id, project_id=project_id,
                                      name=team.name.lower().replace(' ', '-') + '-app', api_key=team.api_key))
            s.flush()
            for user in s.exec(select(User)).all():
                if user.id in migrated_users and user.team_id and s.get(Team, user.team_id):
                    for model in (TeamMember(user_id=user.id, team_id=user.team_id),
                                  ProjectMember(user_id=user.id, project_id='project-' + user.team_id),
                                  ApplicationMember(user_id=user.id, application_id='app-' + user.team_id)):
                        # Composite primary keys make insertion idempotent.
                        keys = (model.user_id, getattr(model, 'team_id', None) or getattr(model, 'project_id', None) or model.application_id)
                        if not s.get(type(model), keys):
                            s.add(model)
            # Existing seeded entries have known team assignments; retain live history untouched.
            for row in s.exec(select(Usage).where(Usage.source == 'seeded')).all():
                if row.team_id and not row.application_id:
                    row.project_id, row.application_id = 'project-' + row.team_id, 'app-' + row.team_id
                    s.add(row)
            s.commit()

    def migrate_lambda(self):
        """Rename the 0–1 alpha settings to lambda without changing their values."""
        tables = set(inspect(self.engine).get_table_names())
        pairs = {'user': [('alpha_preference','lam_preference')],
                 'team': [('enforced_alpha','enforced_lam')],
                 'project': [('configured_alpha','configured_lam'),('alpha_locked','lam_locked')],
                 'application': [('enforced_alpha','enforced_lam'),('alpha_locked','lam_locked')]}
        with self.engine.begin() as connection:
            for table, names in pairs.items():
                if table not in tables:
                    continue
                columns = {c['name'] for c in inspect(connection).get_columns(table)}
                for old, new in names:
                    if old in columns and new not in columns:
                        connection.exec_driver_sql(f'ALTER TABLE "{table}" RENAME COLUMN "{old}" TO "{new}"')
            if 'account' in tables:
                for account_id, raw in connection.exec_driver_sql('SELECT id, settings FROM account').all():
                    settings = json.loads(raw)
                    if 'alpha' in settings:
                        settings['lam'] = settings.pop('alpha')
                        settings['enforce_lam'] = settings.pop('enforce_alpha', False)
                        connection.exec_driver_sql('UPDATE account SET settings=? WHERE id=?',
                                                   (json.dumps(settings), account_id))

    def for_account(self, customer_id):
        scoped = object.__new__(Store)
        scoped.directory, scoped.engine, scoped.customer_id = self.directory, self.engine, customer_id
        return scoped

    def seed_users(self):
        seed = json.loads((ROOT / 'data/demo/seed.json').read_text())
        with Session(self.engine) as s:
            personal = seed['private_account']
            if not s.get(Account, personal['id']):
                s.add(Account(**personal, api_key='lonn_private_' + secrets.token_urlsafe(32),
                              settings=self.account().settings))
            for item in seed['users']:
                if not s.get(User, item['id']):
                    fields = {k: v for k, v in item.items() if k != 'password'}
                    s.add(User(**fields, password_hash=password_hash(item['password'])))
            s.commit()

    def users(self):
        with Session(self.engine) as s:
            return s.exec(select(User).where(User.customer_id == self.customer_id)).all()

    def user(self, user_id=None, email=None):
        with Session(self.engine) as s:
            return s.get(User, user_id) if user_id else s.exec(select(User).where(User.email == email)).first()

    def accounts(self):
        with Session(self.engine) as s:
            return s.exec(select(Account)).all()

    def create_user(self, *, name, email, password, team_id):
        with Session(self.engine) as s:
            user = User(id=str(uuid4()), customer_id=self.customer_id, name=name, email=email,
                        password_hash=password_hash(password), role='enterprise_user', team_id=team_id)
            s.add(user)
            s.commit()
            s.refresh(user)
            return user

    def update_user(self, user_id, **values):
        with Session(self.engine) as s:
            user = s.get(User, user_id)
            if user is None or user.customer_id != self.customer_id or user.role != 'enterprise_user':
                return None
            previous_team = user.team_id
            for field in ('team_id', 'enabled'):
                if field in values:
                    setattr(user, field, values[field])
            s.add(user)
            team_id = values.get('team_id')
            if team_id and team_id != previous_team:
                for member in (TeamMember(user_id=user.id, team_id=team_id),
                               ProjectMember(user_id=user.id, project_id='project-' + team_id),
                               ApplicationMember(user_id=user.id, application_id='app-' + team_id)):
                    scope = getattr(member, 'team_id', None) or getattr(member, 'project_id', None) or member.application_id
                    if not s.get(type(member), (user.id, scope)):
                        s.add(member)
            s.commit()
            s.refresh(user)
            return user

    def seed_teams(self):
        seed = json.loads((ROOT / 'data/demo/seed.json').read_text())
        with Session(self.engine) as s:
            for team in seed['teams']:
                if not s.get(Team, team['id']):
                    s.add(Team(**team, customer_id=self.account().id,
                               api_key='lonn_team_' + secrets.token_urlsafe(32)))
            s.commit()

    def teams(self):
        with Session(self.engine) as s:
            return s.exec(select(Team).where(Team.customer_id == self.account().id).order_by(Team.name)).all()

    def create_team(self, name):
        with Session(self.engine) as s:
            team = Team(id=str(uuid4()), customer_id=self.account().id, name=name,
                        api_key='lonn_team_' + secrets.token_urlsafe(32))
            s.add(team)
            s.commit()
            s.refresh(team)
            return team

    def update_team(self, team_id, *, rotate=False, **values):
        with Session(self.engine) as s:
            team = s.get(Team, team_id)
            if team is None or team.customer_id != self.account().id:
                return None
            for field in ('name', 'enabled', 'disabled_models'):
                if field in values:
                    setattr(team, field, json.dumps(values[field]) if field == 'disabled_models' else values[field])
            if rotate:
                team.api_key = 'lonn_team_' + secrets.token_urlsafe(32)
            s.add(team)
            s.commit()
            s.refresh(team)
            return team

    def migrate_seeded_tariff(self):
        # Refresh only synthetic fixtures; real historical invoices are immutable.
        with Session(self.engine) as session:
            rows = session.exec(select(Usage).where(Usage.customer_id == self.customer_id, Usage.source == 'seeded')).all()
            today = datetime.now(timezone.utc).date().isoformat()
            details = (json.loads(row.details) for row in rows)
            outdated = any(item.get('billing', {}).get('service_fee_basis') != FEE_BASIS
                           or item.get('seed_window_end', '')[:10] != today for item in details)
        if outdated:
            self.refresh_seeded_history()

    def refresh_seeded_history(self):
        """Update only synthetic fixtures; retain real usage, credentials and settings."""
        seed = json.loads((ROOT / 'data/demo/seed.json').read_text())
        with Session(self.engine) as s:
            for row in s.exec(select(Usage).where(Usage.customer_id == seed['id'], Usage.source == 'seeded')):
                s.delete(row)
            s.flush()
            for values in history(seed, self.catalog()):
                s.add(Usage(**values))
            s.commit()

    def account(self):
        with Session(self.engine) as s:
            return s.get(Account, self.customer_id)

    def settings(self, value=None):
        with Session(self.engine) as s:
            account = s.get(Account, self.account().id)
            if value is not None:
                account.settings = json.dumps(normalize_settings(value))
                s.add(account)
                s.commit()
            return normalize_settings(json.loads(account.settings))

    def rotate(self):
        with Session(self.engine) as s:
            account = s.get(Account, self.account().id)
            account.api_key = 'lonn_demo_' + secrets.token_urlsafe(32)
            s.add(account)
            s.commit()
            s.refresh(account)
            return account.api_key

    def catalog(self):
        with Session(self.engine) as s:
            entries = [json.loads(row.metadata_json) for row in s.exec(select(Catalog)).all()]
            return [entry for entry in entries if entry.get('active') and entry['id'] in DEMO_SECURITY]

    def record(self, **values):
        with Session(self.engine) as s:
            s.add(Usage(customer_id=self.account().id, **values))
            s.commit()

    def history(self):
        with Session(self.engine) as s:
            return [dict(row.model_dump(), details=json.loads(row.details)) for row in
                    s.exec(select(Usage).where(Usage.customer_id == self.account().id)
                           .order_by(Usage.timestamp.desc())).all()]

    def credential(self):
        path = self.directory / 'provider.key'
        return path.read_text().strip() if path.exists() else ''

    def save_credential(self, key):
        # Restrictive permissions from creation, atomic replacement, no environment mutation.
        path = self.directory / ('provider-' + secrets.token_hex(8) + '.tmp')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write(key)
        path.replace(self.directory / 'provider.key')

    def reset(self):
        SQLModel.metadata.drop_all(self.engine)
        SQLModel.metadata.create_all(self.engine)
        self.seed()
        self.seed_teams()
        self.seed_users()
        self.seed_hierarchy()
