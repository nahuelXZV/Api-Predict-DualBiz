def build_sqlserver_connection_string(params: dict) -> str:
    driver = str(params.get("driver") or "").strip()
    server = str(params.get("server") or "").strip()
    database = str(params.get("database") or "").strip()
    user = str(params.get("user") or params.get("uid") or "").strip()
    password = str(params.get("password") or params.get("pwd") or "").strip()
    port = str(params.get("port") or "").strip()
    trusted_connection = str(params.get("trusted_connection") or "").strip().lower()
    encrypt = str(params.get("encrypt") or "").strip()
    trust_server_certificate = str(params.get("trust_server_certificate") or "").strip()

    missing = [
        name
        for name, value in (
            ("driver", driver),
            ("server", server),
            ("database", database),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "La conexion SQL Server requiere los parametros: "
            + ", ".join(missing)
            + "."
        )

    if port:
        server = f"{server},{port}"

    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
    ]

    if trusted_connection in {"true", "1", "yes", "y", "sspi"}:
        parts.append("Trusted_Connection=yes")
    else:
        if not user or not password:
            raise ValueError(
                "La conexion SQL Server requiere 'user' y 'password', "
                "o bien 'trusted_connection=true'."
            )
        parts.extend([f"UID={user}", f"PWD={password}"])

    if encrypt:
        parts.append(f"Encrypt={encrypt}")
    if trust_server_certificate:
        parts.append(f"TrustServerCertificate={trust_server_certificate}")

    return ";".join(parts)
