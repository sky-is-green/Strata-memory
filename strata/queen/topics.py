"""Domain-vocabulary topics for the system's own labeling tooling.

The corpus generator and the labelers share one topic table; it lives here
(inside the system, not the bench) so ``strata/`` never imports from
``hivebench/`` — the system must be installable and importable on its own.
"""

TOPICS = {
    "authentication": {
        "feature": "auth service",
        "aspects": ["JWT expiry", "refresh-token rotation", "session store", "OAuth2 scopes", "bcrypt hashing"],
        "decisions": {
            "JWT expiry": "15 minutes",
            "refresh-token rotation": "rotate on every refresh",
            "session store": "Redis with TTL",
            "OAuth2 scopes": "read/write split",
            "bcrypt hashing": "cost factor 12",
        },
    },
    "database_schema": {
        "feature": "order schema",
        "aspects": ["normalization", "indexes", "soft deletes", "migrations", "foreign keys"],
        "decisions": {
            "normalization": "3NF with a denormalized read model",
            "indexes": "composite on (customer_id, created_at)",
            "soft deletes": "deleted_at column",
            "migrations": "Alembic, forward-only",
            "foreign keys": "ON DELETE CASCADE",
        },
    },
    "logging": {
        "feature": "log pipeline",
        "aspects": ["structured logs", "log levels", "sampling", "correlation ids", "retention"],
        "decisions": {
            "structured logs": "JSON via python logging",
            "log levels": "INFO by default, DEBUG in dev",
            "sampling": "10% trace sampling",
            "correlation ids": "X-Request-Id header",
            "retention": "30 days hot, 12 months cold",
        },
    },
    "deployment": {
        "feature": "deploy pipeline",
        "aspects": ["blue-green", "health checks", "rollbacks", "canary", "secrets"],
        "decisions": {
            "blue-green": "two live slots",
            "health checks": "/healthz with DB ping",
            "rollbacks": "auto on 5% error rate",
            "canary": "5% for 10 minutes",
            "secrets": "Vault, rotated monthly",
        },
    },
    "api_design": {
        "feature": "REST API",
        "aspects": ["pagination", "versioning", "rate limits", "error envelope", "idempotency"],
        "decisions": {
            "pagination": "cursor-based",
            "versioning": "URL /v1 prefix",
            "rate limits": "100 req/min per key",
            "error envelope": "problem+json",
            "idempotency": "Idempotency-Key header",
        },
    },
}