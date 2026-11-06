"""Small organization hierarchy and key/policy lookup for the local demo."""
import secrets
from uuid import uuid4
from sqlmodel import Field, SQLModel, Session, select
from lonn.task_router import DEFAULT_LAMBDA


class Project(SQLModel, table=True):
    id: str = Field(primary_key=True)
    customer_id: str = Field(index=True)
    team_id: str = Field(index=True)
    name: str
    configured_lam: float | None = None
    lam_locked: bool = False


class Application(SQLModel, table=True):
    id: str = Field(primary_key=True)
    customer_id: str = Field(index=True)
    project_id: str = Field(index=True)
    name: str
    api_key: str
    enforced_lam: float | None = None
    lam_locked: bool = True  # Existing enforced_lam values were always locked.
    enabled: bool = True
    security_overrides: str = '{}'


class TeamMember(SQLModel, table=True):
    user_id: str = Field(primary_key=True)
    team_id: str = Field(primary_key=True)
    role: str = 'member'


class ProjectMember(SQLModel, table=True):
    user_id: str = Field(primary_key=True)
    project_id: str = Field(primary_key=True)


class ApplicationMember(SQLModel, table=True):
    user_id: str = Field(primary_key=True)
    application_id: str = Field(primary_key=True)
    role: str = 'member'


def new_id():
    return str(uuid4())


def new_key(kind):
    return f'lonn_{kind}_' + secrets.token_urlsafe(32)


def application_context(store, reference):
    with Session(store.engine) as s:
        apps = s.exec(select(Application).where(Application.customer_id == store.customer_id)).all()
        matches = [a for a in apps if a.id == reference or a.name == reference]
        if len(matches) != 1:
            return None
        app = matches[0]
        project = s.get(Project, app.project_id)
        from .store import Team
        team = s.get(Team, project.team_id) if project else None
        if not team or team.customer_id != store.customer_id or project.customer_id != store.customer_id:
            return None
        return app, project, team


def has_access(store, user, app, project, team):
    if user.customer_id != store.customer_id or not user.enabled:
        return False
    if user.role in ('enterprise_admin', 'private_user'):
        return True
    with Session(store.engine) as s:
        return bool(s.get(TeamMember, (user.id, team.id)) and
                    s.get(ProjectMember, (user.id, project.id)) and
                    s.get(ApplicationMember, (user.id, app.id)))


def lambda_state(store, team, project=None, app=None, user=None):
    """Enforced ancestors win; unlocked local values are inheritable defaults."""
    settings = store.settings()
    if settings.get('enforce_lam', False):
        return {'lam': settings['lam'], 'source': 'organization', 'locked': True}
    if team.enforced_lam is not None:
        return {'lam': team.enforced_lam, 'source': 'team', 'locked': True}
    if project and project.configured_lam is not None and project.lam_locked:
        return {'lam': project.configured_lam, 'source': 'project', 'locked': True}
    if app and app.enforced_lam is not None and app.lam_locked:
        return {'lam': app.enforced_lam, 'source': 'application', 'locked': True}
    if user and user.lam_preference is not None:
        return {'lam': user.lam_preference, 'source': 'user', 'locked': False}
    if app and app.enforced_lam is not None:
        return {'lam': app.enforced_lam, 'source': 'application', 'locked': False}
    if project and project.configured_lam is not None:
        return {'lam': project.configured_lam, 'source': 'project', 'locked': False}
    return {'lam': DEFAULT_LAMBDA, 'source': 'default', 'locked': False}
