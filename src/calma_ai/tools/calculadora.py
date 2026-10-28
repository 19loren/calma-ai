import ast
import operator

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

_OPERADORES = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calcular(expressao: str) -> float:
    """Avalia apenas aritmética básica (+, -, *, /, **, parênteses), sem eval()."""
    texto = expressao.replace(",", ".").replace("x", "*").replace("×", "*").replace("÷", "/")
    arvore = ast.parse(texto, mode="eval")

    def avaliar(no: ast.AST) -> float:
        if isinstance(no, ast.Expression):
            return avaliar(no.body)
        if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)):
            return no.value
        if isinstance(no, ast.BinOp) and type(no.op) in _OPERADORES:
            esquerda, direita = avaliar(no.left), avaliar(no.right)
            if isinstance(no.op, ast.Pow) and abs(direita) > 10:
                raise ValueError("Expoente muito grande.")
            return _OPERADORES[type(no.op)](esquerda, direita)
        if isinstance(no, ast.UnaryOp) and type(no.op) in _OPERADORES:
            return _OPERADORES[type(no.op)](avaliar(no.operand))
        raise ValueError(f"Elemento não permitido na expressão: {ast.dump(no)}")

    return avaliar(arvore)


class CalculadoraInput(BaseModel):
    expressao: str = Field(
        description="Expressão aritmética, ex.: '180 * 10 / 100' ou '(1200 - 300) / 3'. "
        "Para porcentagem escreva a conta completa (não use o símbolo %)."
    )


class CalculadoraTool(BaseTool):
    name: str = "calculadora"
    description: str = (
        "Calcula expressões aritméticas com precisão. Use para descobrir a resposta "
        "correta do exercício e para conferir as contas que o estudante fez. "
        "O resultado é para uso interno: não o revele ao estudante."
    )
    args_schema: type[BaseModel] = CalculadoraInput

    def _run(self, expressao: str) -> str:
        try:
            resultado = calcular(expressao)
        except ZeroDivisionError:
            return "Erro: divisão por zero."
        except (SyntaxError, ValueError, TypeError) as erro:
            return f"Erro: expressão inválida ({erro})."
        if isinstance(resultado, float) and resultado.is_integer():
            resultado = int(resultado)
        elif isinstance(resultado, float):
            resultado = round(resultado, 6)
        return f"{expressao} = {resultado}"
