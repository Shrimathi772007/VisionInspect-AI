"""Create the first quality engineer (or promote an existing user) in the configured database.

Self-registration always creates a factory_supervisor, so this is how the first quality
engineer is made. Later ones can be promoted by a quality engineer via PATCH /users/{id}/role.

Usage (from backend/):
    python scripts/create_quality_engineer.py --name "Jane Doe" --email jane@example.com
    python scripts/create_quality_engineer.py --email existing@example.com --promote-existing

The password is never accepted as an argument. It is read from the environment variable
VISIONINSPECT_BOOTSTRAP_PASSWORD if that is set, otherwise prompted for twice (hidden). It is
never printed or logged. Promoting an existing user keeps their password and asks for none.
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth.bootstrap import (  # noqa: E402
    BootstrapError,
    create_or_promote_quality_engineer,
    find_user_by_email,
    validate_password,
)
from app.database import SessionLocal  # noqa: E402

PASSWORD_ENV_VAR = "VISIONINSPECT_BOOTSTRAP_PASSWORD"


def read_password() -> str:
    password = os.environ.get(PASSWORD_ENV_VAR)
    if password is not None:
        print(f"Using the password from {PASSWORD_ENV_VAR}.")
        validate_password(password)
        return password

    password = getpass.getpass("Password: ")
    validate_password(password)
    if getpass.getpass("Confirm password: ") != password:
        raise BootstrapError("Passwords do not match")
    return password


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or promote a quality engineer account.")
    parser.add_argument("--email", required=True, help="Email address of the account")
    parser.add_argument("--name", help="Display name (required when creating a new account)")
    parser.add_argument(
        "--promote-existing",
        action="store_true",
        help="If the email already exists, change that user's role to quality_engineer",
    )
    args = parser.parse_args()

    with SessionLocal() as db:
        try:
            existing_user = find_user_by_email(db, args.email)
            password = None
            if existing_user is None:
                if not args.name:
                    raise BootstrapError("--name is required when creating a new account")
                password = read_password()
            elif not args.promote_existing:
                raise BootstrapError(
                    f"A user with email {existing_user.email} already exists; "
                    "use --promote-existing to make them a quality engineer"
                )

            user = create_or_promote_quality_engineer(
                db, args.name or "", args.email, password, promote_existing=args.promote_existing
            )
            db.commit()
        except BootstrapError as exc:
            db.rollback()
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        action = "Promoted" if existing_user is not None else "Created"
        print(f"{action} quality engineer: id={user.id} email={user.email} role={user.role.value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
