# Security guidance

Do not commit or publish any of the following:

- AWS access keys or secret access keys
- Database usernames or passwords
- PostgreSQL connection URIs containing passwords
- Kubernetes Secret manifests containing real credentials
- `.env` files
- `.dvc/config.local`

Use environment variables, Kubernetes Secrets, AWS IAM roles, or a managed secrets service. If a credential is exposed, rotate it immediately rather than deleting only the visible line.
