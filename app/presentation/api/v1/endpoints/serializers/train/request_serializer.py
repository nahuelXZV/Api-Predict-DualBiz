from rest_framework import serializers


class DataSourceSerializer(serializers.Serializer):
    type = serializers.ChoiceField(
        choices=["sqlserver", "csv"],
        help_text="Tipo de datasource. Opciones: 'sqlserver', 'csv'.",
    )
    params = serializers.DictField(
        required=False,
        default=dict,
        help_text=(
            "Parámetros opcionales del datasource. "
            "sqlserver: {connection_string, query}. "
            "csv: {path, separator, encoding}."
        ),
    )


class TrainRequestSerializer(serializers.Serializer):
    model_name = serializers.CharField(
        default="pedido_sugerido",
        help_text="Nombre del modelo a entrenar.",
    )
    version = serializers.CharField(
        default="1.0",
        help_text="Versión del modelo.",
    )
    data_source = DataSourceSerializer(
        help_text="Origen de los datos para el entrenamiento.",
    )
