import sys
import warnings
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# =============================================================================
# LOG: escreve na tela E em log_avaliacao.txt ao mesmo tempo
# =============================================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

log_file = open("log_avaliacao.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

# Esconde os avisos de "DeprecationWarning" para o terminal ficar limpo
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. Configurar o Juiz (Qwen 1.5B, mais rápido na CPU)
llm_local = ChatOllama(model="qwen2.5:1.5b", temperature=0)
embeddings_local = OllamaEmbeddings(model="nomic-embed-text")

juiz_llm = LangchainLLMWrapper(llm_local)
juiz_embeddings = LangchainEmbeddingsWrapper(embeddings_local)

# 2. Dataset de teste (30 perguntas)
dados_teste = {
    "question": [
        "O que é uma fração equivalente?",
        "Como resolvo uma equação de primeiro grau?",
        "O que é uma fração própria?",
        "Qual é a diferença entre numerador e denominador?",
        "Como calcular a porcentagem de um número?",
        "O que são números primos?",
        "Como calcular a área de um retângulo?",
        "O que é uma equação matemática?",
        "Como calcular a média aritmética?",
        "Qual é a diferença entre perímetro e área?",
        "O que é uma potência?",
        "Como funciona a regra de sinais na multiplicação?",
        "O que são números inteiros?",
        "Como transformar uma fração em número decimal?",
        "O que é uma raiz quadrada?",
        "Como calcular o perímetro de um quadrado?",
        "O que é uma expressão numérica?",
        "Como resolver uma equação com incógnita dos dois lados?",
        "O que é um múltiplo de um número?",
        "O que é um divisor de um número?",
        "Como calcular o volume de um cubo?",
        "O que é uma proporção?",
        "Como calcular juros simples?",
        "Qual é a diferença entre números naturais e números inteiros?",
        "O que é uma função matemática?",
        "Como identificar um triângulo retângulo?",
        "O que é o mínimo múltiplo comum (MMC)?",
        "Como calcular o máximo divisor comum (MDC)?",
        "O que é uma sequência numérica?",
        "Como resolver um problema matemático utilizando a interpretação do enunciado?"
    ],
    "answer": [
        "São frações que representam a mesma quantidade, como 1/2 e 2/4.",
        "É necessário isolar o x, aplicando operações inversas nos dois lados da igualdade.",
        "É uma fração em que o numerador é menor que o denominador.",
        "O numerador indica quantas partes foram consideradas e o denominador indica em quantas partes iguais o todo foi dividido.",
        "Multiplicamos o número pela porcentagem desejada e dividimos o resultado por 100.",
        "São números naturais maiores que 1 que possuem exatamente dois divisores positivos: 1 e eles mesmos.",
        "Multiplicamos a medida da base pela medida da altura.",
        "É uma igualdade matemática que contém uma ou mais incógnitas.",
        "Somamos todos os valores e dividimos o resultado pela quantidade de valores.",
        "O perímetro é a medida do contorno de uma figura, enquanto a área representa a medida de sua superfície.",
        "É uma operação matemática que representa a multiplicação de um número por ele mesmo várias vezes.",
        "Na multiplicação, sinais iguais resultam em positivo e sinais diferentes resultam em negativo.",
        "São números positivos, negativos e o zero, sem considerar partes decimais.",
        "Basta dividir o numerador pelo denominador.",
        "É um número que, multiplicado por ele mesmo, resulta no número original.",
        "Somamos os quatro lados do quadrado, ou multiplicamos a medida de um lado por quatro.",
        "É uma combinação de números e operações matemáticas que deve ser resolvida seguindo a ordem correta das operações.",
        "Agrupamos os termos com incógnita em um lado e os termos independentes no outro, utilizando operações equivalentes.",
        "É qualquer número que pode ser obtido pela multiplicação desse número por um número inteiro.",
        "É um número que divide outro exatamente, sem deixar resto.",
        "Multiplicamos a medida da aresta por ela mesma três vezes.",
        "É uma relação de igualdade entre duas razões.",
        "Multiplicamos o capital pela taxa de juros e pelo tempo, utilizando a taxa na forma decimal.",
        "Os naturais incluem zero e os números positivos de contagem, enquanto os inteiros também incluem os negativos.",
        "É uma relação que associa cada elemento de um conjunto de entrada a exatamente um elemento de um conjunto de saída.",
        "É um triângulo que possui um ângulo de 90 graus.",
        "É o menor múltiplo positivo comum entre dois ou mais números inteiros.",
        "É o maior número inteiro positivo que divide dois ou mais números sem deixar resto.",
        "É uma lista de números organizada segundo uma regra ou padrão.",
        "Devemos identificar os dados fornecidos, compreender o que está sendo solicitado, escolher as operações adequadas e resolver o problema."
    ],
    "contexts": [
        ["Fração equivalente: frações que representam a mesma parte de um todo, obtidas multiplicando ou dividindo numerador e denominador pelo mesmo número."],
        ["Equação de primeiro grau: igualdade matemática com incógnita de expoente um. Sua resolução envolve isolar a incógnita utilizando operações inversas."],
        ["Fração própria: fração cujo numerador é menor que o denominador, representando uma quantidade menor que um inteiro."],
        ["Numerador é o número que fica acima do traço da fração e representa as partes consideradas. Denominador fica abaixo e representa a quantidade de partes iguais do todo."],
        ["Porcentagem representa uma razão de denominador 100. Para calcular uma porcentagem, multiplicamos o valor pela taxa percentual e dividimos por 100."],
        ["Números primos são números naturais maiores que 1 que possuem exatamente dois divisores positivos: o número 1 e ele próprio."],
        ["A área de um retângulo é calculada multiplicando sua base por sua altura. A = base × altura."],
        ["Uma equação é uma igualdade matemática entre duas expressões que pode conter valores desconhecidos chamados incógnitas."],
        ["A média aritmética simples é calculada pela soma de todos os valores dividida pela quantidade de elementos."],
        ["Perímetro é a soma dos comprimentos dos lados de uma figura. Área é a medida da superfície ocupada por essa figura."],
        ["Potenciação representa uma multiplicação de fatores iguais. A base é o fator repetido e o expoente indica quantas vezes ele aparece."],
        ["Na multiplicação de números inteiros, sinais iguais produzem resultado positivo e sinais diferentes produzem resultado negativo."],
        ["Números inteiros são formados pelos números negativos, pelo zero e pelos números positivos, sem parte decimal."],
        ["Para converter uma fração em decimal, dividimos o numerador pelo denominador."],
        ["A raiz quadrada de um número não negativo é o valor que, elevado ao quadrado, resulta nesse número."],
        ["O perímetro de um quadrado é calculado multiplicando a medida de um lado por quatro, pois seus quatro lados possuem a mesma medida."],
        ["Expressões numéricas combinam números e operações. A resolução respeita a ordem: parênteses, potências e raízes, multiplicações e divisões, adições e subtrações."],
        ["Em equações com incógnitas nos dois lados, agrupamos os termos com incógnita em um lado e os termos constantes no outro, realizando operações equivalentes nos dois membros."],
        ["Um múltiplo de um número é o resultado da multiplicação desse número por um número inteiro."],
        ["Um divisor é um número que divide outro número sem deixar resto."],
        ["O volume de um cubo é calculado elevando a medida de sua aresta ao cubo: V = lado × lado × lado."],
        ["Uma proporção é a igualdade entre duas razões. Ela permite comparar relações entre grandezas."],
        ["Juros simples são calculados pela fórmula J = C × i × t, em que C é o capital, i é a taxa de juros e t é o tempo."],
        ["Números naturais são os números de contagem, incluindo zero na convenção usual. Números inteiros incluem também os números negativos."],
        ["Uma função matemática é uma relação que associa cada elemento do domínio a exatamente um elemento do contradomínio."],
        ["Um triângulo retângulo é aquele que possui um ângulo interno de 90 graus. O lado oposto a esse ângulo é chamado hipotenusa."],
        ["O mínimo múltiplo comum é o menor múltiplo positivo compartilhado por dois ou mais números."],
        ["O máximo divisor comum é o maior número inteiro positivo que divide dois ou mais números sem deixar resto."],
        ["Uma sequência numérica é uma lista ordenada de números que segue uma determinada regra de formação."],
        ["A resolução de problemas matemáticos envolve interpretar o enunciado, identificar os dados e a pergunta, selecionar estratégias e operações e verificar o resultado."]
    ],
    "ground_truth": [
        "Frações equivalentes representam a mesma quantidade ou parte de um todo.",
        "Resolver uma equação de primeiro grau significa encontrar o valor da incógnita, isolando-a por meio de operações inversas.",
        "É uma fração cujo numerador é menor que o denominador.",
        "O numerador representa as partes consideradas e o denominador representa o total de partes iguais.",
        "Multiplicamos o valor pela porcentagem e dividimos por 100.",
        "São números naturais maiores que 1 com exatamente dois divisores positivos.",
        "Multiplicamos a base pela altura.",
        "É uma igualdade matemática que pode conter incógnitas.",
        "Somamos os valores e dividimos pela quantidade de valores.",
        "Perímetro mede o contorno e área mede a superfície.",
        "É a multiplicação repetida de uma base por ela mesma.",
        "Sinais iguais resultam em positivo e sinais diferentes resultam em negativo.",
        "São os números negativos, o zero e os positivos, sem partes decimais.",
        "Dividimos o numerador pelo denominador.",
        "É o número que elevado ao quadrado resulta no número original.",
        "Multiplicamos a medida do lado por quatro.",
        "É uma combinação de números e operações matemáticas.",
        "Agrupamos as incógnitas em um lado e os termos independentes no outro.",
        "É o resultado da multiplicação de um número por um inteiro.",
        "É um número que divide outro sem deixar resto.",
        "Multiplicamos a aresta por ela mesma três vezes.",
        "É a igualdade entre duas razões.",
        "Juros simples são calculados por J = C × i × t.",
        "Os naturais incluem os números de contagem e os inteiros incluem também os negativos.",
        "É uma relação que associa cada entrada a exatamente uma saída.",
        "É um triângulo que possui um ângulo de 90 graus.",
        "É o menor múltiplo positivo comum entre dois ou mais números.",
        "É o maior divisor comum entre dois ou mais números.",
        "É uma lista ordenada de números que segue uma regra.",
        "É necessário interpretar os dados, identificar o que se pede, escolher as operações e resolver."
    ]
}

dataset_ragas = Dataset.from_dict(dados_teste)

# 3. Configurar Timeout e Concorrência
run_config = RunConfig(timeout=300, max_workers=1, max_retries=2)

print("Iniciando a avaliação com RAGAS. Isso pode demorar alguns minutos no Codespace...")

# 4. Executar a avaliação
resultado = evaluate(
    dataset=dataset_ragas,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=juiz_llm,
    embeddings=juiz_embeddings,
    run_config=run_config
)

# 5. Mostrar os resultados
df_metricas = resultado.to_pandas().reset_index(drop=True)
df_dataset  = dataset_ragas.to_pandas().reset_index(drop=True)   # ← nome certo
df_final    = pd.concat([df_dataset, df_metricas], axis=1)

print("--- RESULTADO FINAL ---")
print(resultado)

print("\n--- DETALHAMENTO POR PERGUNTA ---")
print(df_final[["question", "faithfulness", "answer_relevancy",
                "context_precision", "context_recall"]].to_string(index=False))

df_final.to_csv("resultados_ragas.csv", index=False)
print("\nSalvo em resultados_ragas.csv")
print("Log salvo em log_avaliacao.txt")

# Fecha o arquivo de log
log_file.close()