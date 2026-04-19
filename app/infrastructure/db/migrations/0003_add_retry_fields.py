import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_initial'),
    ]

    operations = [
        # FK nullable: no tiene default constraint, AddField directo funciona
        migrations.AddField(
            model_name='ejecuciontareaprogramada',
            name='ejecucion_original',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='reintentos',
                to='app.ejecuciontareaprogramada',
            ),
        ),
        # numero_intento y max_reintentos tienen default NOT NULL.
        # mssql-django intenta DROP CONSTRAINT <nombre_campo> pero SQL Server usa
        # nombres autogenerados → falla. Usamos SeparateDatabaseAndState para
        # ejecutar ALTER TABLE directo y mantener el estado de Django sincronizado.
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunSQL(
                    sql="ALTER TABLE [ml].[ejecucion_tarea_programada] ADD numero_intento smallint NOT NULL DEFAULT 1",
                    reverse_sql="ALTER TABLE [ml].[ejecucion_tarea_programada] DROP COLUMN numero_intento",
                ),
            ],
            state_operations=[
                migrations.AddField(
                    model_name='ejecuciontareaprogramada',
                    name='numero_intento',
                    field=models.PositiveSmallIntegerField(default=1),
                ),
            ],
        ),
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunSQL(
                    sql="ALTER TABLE [ml].[tarea_programada] ADD max_reintentos smallint NOT NULL DEFAULT 0",
                    reverse_sql="ALTER TABLE [ml].[tarea_programada] DROP COLUMN max_reintentos",
                ),
            ],
            state_operations=[
                migrations.AddField(
                    model_name='tareaprogramada',
                    name='max_reintentos',
                    field=models.PositiveSmallIntegerField(default=0),
                ),
            ],
        ),
    ]
