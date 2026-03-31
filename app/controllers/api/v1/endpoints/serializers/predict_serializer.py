from rest_framework import serializers


class PredictParametersSerializer(serializers.Serializer):
    cliente_id = serializers.IntegerField(
        default=14111,
        help_text="ID del cliente para el que se generan las sugerencias.",
    )
    cantidad_minima = serializers.FloatField(
        default=1.0,
        help_text="Cantidad mínima a sugerir por producto.",
    )
    top_n = serializers.IntegerField(
        default=50,
        help_text="Número máximo de productos a retornar.",
    )
    solo_nuevos = serializers.BooleanField(
        default=False,
        help_text="Si es true, solo retorna productos que el cliente nunca compró.",
    )
    porcentaje_pareto = serializers.IntegerField(
        default=20,
        min_value=1,
        max_value=100,
        help_text="Porcentaje de productos a conservar aplicando el filtro de Pareto (1-100).",
    )


class PredictRequestSerializer(serializers.Serializer):
    model_name = serializers.CharField(
        default="pedido_sugerido",
        help_text="Nombre del modelo a usar para la predicción.",
    )
    parameters = PredictParametersSerializer()
