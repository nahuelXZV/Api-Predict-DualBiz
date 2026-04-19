from django.db import migrations


class Migration(migrations.Migration):
    initial = True
    dependencies = []

    operations = [
        migrations.RunSQL(
            "IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'ml') EXEC('CREATE SCHEMA ml')",
            reverse_sql="DROP SCHEMA ml",
        ),
    ]
