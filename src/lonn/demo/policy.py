"""Fail closed for unknown privacy guarantees; never simulate compliance."""
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enforce_lam: bool = False
    lam: float = Field(default=0.20, ge=0, le=1, allow_inf_nan=False, strict=True)
    require_zdr: bool = False
    forbid_training: bool = False
    region: str = Field(default='any', pattern='^(any|EU|US)$')
    max_retention_days: int | None = Field(default=None, ge=0, le=3650)
    disabled_models: list[str] = Field(default_factory=list)


def catalog_status(entry, settings, table):
    reasons = []
    if not entry['active']:
        reasons.append('Inactive in prototype catalog')
    if entry['provider'] != 'openrouter':
        reasons.append('Demo supports OpenRouter only')
    if entry['id'] not in table.global_row:
        reasons.append('No routing-table measurements')
    if entry['id'] in settings['disabled_models']:
        reasons.append('Disabled by account')
    # Existing catalog has no verified endpoint-level compliance information.
    # Model publisher identity cannot establish the serving endpoint's policy.
    for enabled, label in [(settings['require_zdr'], 'Zero data retention'),
                           (settings['forbid_training'], 'No training'),
                           (settings['region'] != 'any', 'Processing region'),
                           (settings['max_retention_days'] is not None, 'Retention limit')]:
        if enabled:
            reasons.append(f'{label}: unknown / unverified')
    return dict(entry, eligible=not reasons, exclusion_reasons=reasons,
                privacy_status='Unknown — no verified endpoint metadata',
                availability='Configured, not live-verified' if entry['active'] else 'Inactive')
