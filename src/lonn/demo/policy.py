"""Apply the customer demo's assumed model security and region metadata."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from lonn.task_router import DEFAULT_LAMBDA


REGIONS = ('north_america', 'south_america', 'europe', 'asia', 'africa', 'oceania')
# Lambda is a quality preference from cost only (0) to quality only (1).
MAX_DEMO_LAMBDA = 1.0
Region = Literal['north_america', 'south_america', 'europe', 'asia', 'africa', 'oceania']
REQUIREMENTS = (
    ('require_zdr', 'zero_data_retention', 'Zero data retention'),
    ('forbid_training', 'no_training', 'No training'),
    ('require_encryption', 'encryption', 'Encryption'),
    ('require_private_network', 'private_network', 'Private network'),
    ('require_audit_logs', 'audit_logs', 'Audit logs'),
)
SECURITY_PATH = Path(__file__).resolve().parents[3] / 'config' / 'demo_security.yaml'
DEMO_SECURITY = yaml.safe_load(SECURITY_PATH.read_text())
for model_id, metadata in DEMO_SECURITY.items():
    capabilities = {key for _, key, _ in REQUIREMENTS}
    if (not isinstance(metadata, dict) or set(metadata) != capabilities | {'regions'}
            or not isinstance(metadata['regions'], list) or not metadata['regions']
            or not set(metadata['regions']) <= set(REGIONS)
            or any(type(metadata[key]) is not bool for key in capabilities)):
        raise ValueError(f'Invalid demo security metadata for {model_id}')


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enforce_lam: bool = False
    lam: float = Field(default=DEFAULT_LAMBDA, ge=0, le=MAX_DEMO_LAMBDA, allow_inf_nan=False, strict=True)
    require_zdr: bool = False
    forbid_training: bool = False
    require_encryption: bool = False
    require_private_network: bool = False
    require_audit_logs: bool = False
    allowed_regions: list[Region] = Field(default_factory=lambda: list(REGIONS), max_length=len(REGIONS))
    max_retention_days: int | None = Field(default=None, ge=0, le=3650)
    disabled_models: list[str] = Field(default_factory=list)

    @field_validator('allowed_regions')
    @classmethod
    def unique_regions(cls, regions):
        if len(regions) != len(set(regions)):
            raise ValueError('Each region may appear only once')
        return regions


class ApplicationSecurity(BaseModel):
    """Application restrictions can only add to the organization's policy."""
    model_config = ConfigDict(extra='forbid')
    require_zdr: bool = False
    forbid_training: bool = False
    require_encryption: bool = False
    require_private_network: bool = False
    require_audit_logs: bool = False
    allowed_regions: list[Region] | None = Field(default=None, max_length=len(REGIONS))

    @field_validator('allowed_regions')
    @classmethod
    def unique_regions(cls, regions):
        if regions is not None and len(regions) != len(set(regions)):
            raise ValueError('Each region may appear only once')
        return regions


def application_security(settings, overrides):
    """Combine organization policy and app restrictions without weakening either."""
    values = ApplicationSecurity.model_validate(overrides).model_dump()
    effective = dict(settings)
    for key, _, _ in REQUIREMENTS:
        effective[key] = settings[key] or values[key]
    if values['allowed_regions'] is not None:
        effective['allowed_regions'] = [region for region in settings['allowed_regions']
                                        if region in values['allowed_regions']]
    return effective


def normalize_settings(value):
    """Keep legacy restricted regions fail-closed until a continent is selected."""
    data = dict(value)
    old_region = data.pop('region', 'any')
    if 'allowed_regions' not in data:
        data['allowed_regions'] = list(REGIONS) if old_region == 'any' else []
    data['disabled_models'] = [model for model in data.get('disabled_models', []) if model in DEMO_SECURITY]
    return Settings.model_validate(data).model_dump()


def catalog_status(entry, settings, table):
    policy_reasons = []
    security = DEMO_SECURITY.get(entry['id'])
    if not entry['active'] or entry['provider'] != 'openrouter':
        policy_reasons.append('Model is not active in the demo')
    if entry['id'] not in table.global_row:
        policy_reasons.append('No routing-table measurements')
    matching_regions = [region for region in settings['allowed_regions']
                        if security and region in security['regions']]
    if not matching_regions:
        policy_reasons.append('No allowed region supported')
    for setting, capability, label in REQUIREMENTS:
        if settings[setting] and (not security or not security[capability]):
            policy_reasons.append(f'{label} not supported in demo metadata')
    if settings['max_retention_days'] is not None:
        policy_reasons.append('Maximum retention period is unknown')
    reasons = policy_reasons + (['Disabled by account'] if entry['id'] in settings['disabled_models'] else [])
    return dict(entry, eligible=not reasons, exclusion_reasons=reasons,
                policy_eligible=not policy_reasons, policy_exclusion_reasons=policy_reasons,
                task_aptitudes=table.model_aptitudes(entry['id']),
                security=security, matching_regions=matching_regions,
                privacy_status='Demo assumptions, not verified provider guarantees',
                availability='Configured, not live-verified')
