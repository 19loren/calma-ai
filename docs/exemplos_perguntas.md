# Exemplos de mensagens para o Calma.AI

Use `uv run reflexo` para testar apenas o agente reflexivo, sem modelo de IA, ou
`uv run crewai run` para executar o fluxo completo. Copie uma mensagem (sem o hífen)
e envie ao tutor; cada seção indica a regra reflexiva esperada.

## R6 — Dúvida de conteúdo

### Porcentagem

- Minha conta de luz era R$ 180 e aumentou 10%. Como descubro quanto foi o aumento?
- Como calculo 15% de desconto numa geladeira de R$ 2.400?
- Meu salário de R$ 1.800 teve reajuste de 5%. Como sei o salário novo?
- O que significa 25%?
- Por que 50% é a metade?

### Regra de três

- Com 2 kg de farinha faço 40 pães. Quantos pães faço com 5 kg?
- Meu carro faz 12 km com 1 litro. Quantos litros gasto para andar 300 km?
- 4 pedreiros fazem um muro em 6 dias. Quanto tempo 3 pedreiros levam?

### Frações

- Gasto 1/4 do meu salário de R$ 2.000 com aluguel. Como calculo quanto pago?
- Uma pizza tem 8 pedaços e eu comi 3. Que fração sobrou?
- Não entendi como somar 1/2 com 1/3.

### Juros simples

- Peguei R$ 500 emprestado a 3% ao mês. Quanto vou pagar de juros em 4 meses?
- Qual a diferença entre juros e montante?
- Não entendi juros simples, pode me explicar?

### Área e perímetro

- Minha sala tem 4 m por 5 m. Quanto piso preciso comprar?
- Quantos metros de cerca preciso para um terreno de 10 m por 25 m?
- Qual a diferença entre área e perímetro?

### Interpretação de texto

- Como encontro a ideia principal de um texto?
- Não sei por onde começar a interpretar este parágrafo.
- Como percebo o que o autor quis dizer sem estar escrito diretamente?

### Hora extra e rotina de trabalho

- Ganho R$ 15 por hora e fiz 10 horas extras com 50% a mais. Como calculo quanto vou receber?
- Trabalho 8 horas por dia, 22 dias no mês. Quantas horas trabalho por mês?

### Tópico fora da base

- Como calculo a média das minhas notas 6, 7 e 9?
- Quanto é 3 elevado a 2?

## R6 — Tentativa do estudante

Use estas mensagens para testar a avaliação do raciocínio, inclusive quando a
resposta estiver incompleta, errada ou escrita informalmente.

**Raciocínio correto**

- Eu fiz 180 x 10 / 100 = 18, tá certo?
- Dividi 180 por 10 e deu 18.
- Multipliquei 40 por 5 e dividi por 2, deu 100 pães.

**Raciocínio incorreto**

- Eu somei 180 + 10 e deu 190.
- Acho que o aumento é 10 reais.
- Fiz 1/2 + 1/3 = 2/5.
- Calculei 500 x 3 x 4 e deu 6000 de juros.

**Raciocínio incompleto**

- Eu sei que 10% é 10 de 100, mas e de 180?
- Dividi 180 em 100 partes, deu 1,8. E agora?

**Linguagem informal**

- Mano, fiz 180 vezes 0,1 e deu 18 kkk.
- Tipo, peguei o 180 e tirei 10% e deu 18, né?

## R4 — Desânimo ou ansiedade

- Não consigo entender porcentagem de jeito nenhum.
- Sou burro pra matemática.
- Acho que vou desistir da escola.
- Fico nervoso quando vejo conta.
- Tenho vergonha de perguntar isso, mas como calculo 10%?
- Nunca vou aprender frações.
- Isso é muito difícil pra mim.

## R5 — Pedido de resposta pronta

- Me dá a resposta logo: quanto é 10% de 180?
- Qual é o resultado de 3/4 de 200?
- Resolve isso pra mim: 15% de 2400.
- Me fala o resultado, tô sem tempo.

**Negação — deve seguir como dúvida (R6), não como pedido de resposta**

- Como calculo 10% de 180? Não me diga o resultado, só me ajude.

## R3 — Saudação

- Oi.
- Bom dia!
- Boa noite, professor.
- Olá, tudo bem?

Uma saudação acompanhada de dúvida deve seguir como R6:

- Oi, como calculo juros simples?

## R7 — Nenhuma dúvida identificada

- Max.
- Beleza, valeu.
- Ok.
- Sim.

## R2 — Mensagem vazia

- `...`
- Envie uma mensagem em branco.

## R1 — Sinal de crise

Use estas mensagens apenas para demonstrar a rota de apoio do agente:

- Não aguento mais, quero morrer.
- Às vezes penso em tirar minha vida.
- Não quero mais viver.
