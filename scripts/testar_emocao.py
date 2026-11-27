from calma_ai.tools.emoçao import DetectEmocaoTool

detector = DetectEmocaoTool()

resultado = detector._run(
    estudante_id="001",
    mensagem=(
        "Estou tentando entender, mas já errei "
        "várias vezes e estou ficando desanimado."
    )
)

print(resultado)