from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("app", "0001_create_schema"),
    ]

    operations = [
        migrations.CreateModel(
            name="TareaProgramada",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                ("nombre", models.CharField(max_length=100, unique=True)),
                ("nombre_modelo", models.CharField(max_length=100)),
                ("tipo_job", models.CharField(max_length=30)),
                ("cron_schedule", models.CharField(blank=True, max_length=50, null=True)),
                ("activo", models.BooleanField(default=True)),
                ("configuracion", models.JSONField(default=dict)),
                ("creado_en", models.DateTimeField(auto_now_add=True)),
                ("actualizado_en", models.DateTimeField(auto_now=True)),
            ],
            options={"db_table": "[ml].[tarea_programada]"},
        ),
        migrations.CreateModel(
            name="EjecucionTareaProgramada",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                (
                    "tarea_programada",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="ejecuciones",
                        to="app.tareaprogramada",
                    ),
                ),
                ("disparado_por", models.CharField(max_length=20)),
                ("estado", models.CharField(default="pendiente", max_length=20)),
                ("iniciado_en", models.DateTimeField()),
                ("finalizado_en", models.DateTimeField(blank=True, null=True)),
                ("mensaje_error", models.TextField(blank=True, null=True)),
            ],
            options={"db_table": "[ml].[ejecucion_tarea_programada]"},
        ),
        migrations.CreateModel(
            name="LogTareaProgramada",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                (
                    "ejecucion_tarea_programada",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="logs_steps",
                        to="app.ejecuciontareaprogramada",
                    ),
                ),
                ("nombre_step", models.CharField(max_length=100)),
                ("orden_step", models.PositiveSmallIntegerField()),
                ("estado", models.CharField(max_length=20)),
                ("duracion_segundos", models.FloatField(blank=True, null=True)),
                ("mensaje_error", models.TextField(blank=True, null=True)),
                ("ejecutado_en", models.DateTimeField()),
            ],
            options={"db_table": "[ml].[log_tarea_programada]"},
        ),
        migrations.CreateModel(
            name="VersionModelo",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                (
                    "ejecucion_tarea_programada",
                    models.OneToOneField(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="version_modelo",
                        to="app.ejecuciontareaprogramada",
                    ),
                ),
                ("nombre_modelo", models.CharField(max_length=100)),
                ("version", models.CharField(max_length=50)),
                ("entrenado_en", models.DateTimeField()),
                ("ruta_pkl", models.CharField(max_length=500)),
                ("tipo_fuente_datos", models.CharField(max_length=20)),
                ("cantidad_clientes", models.IntegerField(default=0)),
                ("cantidad_productos", models.IntegerField(default=0)),
                ("hiperparametros", models.JSONField(default=dict)),
                ("activo", models.BooleanField(default=False)),
            ],
            options={"db_table": "[ml].[version_modelo]"},
        ),
        migrations.CreateModel(
            name="MetricaModelo",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                (
                    "version_modelo",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="metricas",
                        to="app.versionmodelo",
                    ),
                ),
                ("nombre_metrica", models.CharField(max_length=100)),
                ("valor_metrica", models.FloatField()),
                ("split", models.CharField(blank=True, max_length=20, null=True)),
                ("calculado_en", models.DateTimeField(auto_now_add=True)),
            ],
            options={"db_table": "[ml].[metrica_modelo]"},
        ),
        migrations.CreateModel(
            name="LotePrediccion",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                ("nombre_modelo", models.CharField(max_length=100, unique=True)),
                (
                    "version_modelo",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="lote_prediccion_activo",
                        to="app.versionmodelo",
                    ),
                ),
                ("generado_en", models.DateTimeField()),
                ("cantidad_clientes", models.IntegerField(default=0)),
                ("cantidad_predicciones", models.IntegerField(default=0)),
                ("parametros", models.JSONField(default=dict)),
                ("estado", models.CharField(default="generando", max_length=20)),
            ],
            options={"db_table": "[ml].[lote_prediccion]"},
        ),
        migrations.CreateModel(
            name="ResultadoPrediccion",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                (
                    "lote_prediccion",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="resultados",
                        to="app.loteprediccion",
                    ),
                ),
                ("cliente_id", models.CharField(max_length=50)),
                ("producto_id", models.CharField(max_length=50)),
                ("fuente", models.CharField(max_length=20)),
                ("cantidad_sugerida", models.FloatField()),
                ("score", models.FloatField()),
                ("posicion", models.PositiveSmallIntegerField()),
            ],
            options={"db_table": "[ml].[resultado_prediccion]"},
        ),
    ]
