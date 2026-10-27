"""
Streamlit interface for the narrative analyzer.
Run with:  python -m streamlit run app_narrativa.py
"""
import json
import random
import re
from collections import Counter

import pandas as pd
import spacy
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Narrative Analyzer", page_icon="📖", layout="wide")


@st.cache_resource
def carregar_modelo():
    return spacy.load("en_core_web_sm")


POS_NOMES = {
    "NOUN": "Noun",
    "VERB": "Verb",
    "ADJ": "Adjective",
    "ADV": "Adverb",
    "CCONJ": "Coordinating conjunction",
    "SCONJ": "Subordinating conjunction",
    "INTJ": "Interjection",
    "DET": "Determiner",
    "ADP": "Preposition",
}


COMO_USAR = {
    "Noun": "Names a person, place, thing or idea. Usually the subject or object of a sentence.\n\nExample: The **dog* found a *key.",
    "Verb": "Shows an action or state. Every sentence needs one.\n\nExample: She **runs* every morning.*",
    "Adjective": "Describes a noun. It usually comes before the noun or after to be.\n\nExample: A **tall* man. The house is *old.",
    "Adverb": "Describes a verb, an adjective or another adverb. Often answers how, when or where.\n\nExample: He spoke **quietly.",
    "Coordinating conjunction": "Joins words, phrases or clauses of equal importance. The main ones are for, and, nor, but, or, yet, so. Put a comma before it when it joins two full clauses.\n\nExample: I was tired, **but* I kept walking.*",
    "Subordinating conjunction": "Starts a dependent clause that cannot stand alone, and links it to a main clause. Examples: because, although, if, when, while, since.\n\nExample: She stayed home **because* it was raining.*",
    "Interjection": "A short exclamation that expresses emotion. It is usually separate from the grammar of the sentence.\n\nExample: **Oh, I forgot!",
    "Determiner": "Comes before a noun to specify it: articles (a, an, the), demonstratives (this, that), possessives (my, your) and quantifiers (some, many).\n\nExample: **The* book is on *my* desk.*",
    "Preposition": "Links a noun or pronoun to another word, showing place, time or direction.\n\nExample: The cat is **on* the table.*",
}


def classes_gramaticais(doc):
    """Each word once (case-insensitive), with its most frequent category."""
    contagem = {}
    for t in doc:
        if t.is_alpha and t.pos_ in POS_NOMES:
            palavra = t.text.lower()
            contagem.setdefault(palavra, Counter())[POS_NOMES[t.pos_]] += 1
    linhas = [
        {"Word": palavra, "Part of Speech": c.most_common(1)[0][0]}
        for palavra, c in contagem.items()
    ]
    return pd.DataFrame(linhas, columns=["Word", "Part of Speech"])


EXEMPLOS_FIXOS = {
    "Coordinating conjunction": {
        "and": "She opened the door and walked in.|Tea and coffee are both ready.",
        "but": "I wanted to go, but it was too late.|He is small but strong.",
        "or": "Do you want tea or coffee?|Hurry up, or we will miss the bus.",
        "nor": "He neither called nor wrote.|She did not eat, nor did she drink.",
        "for": "He stayed home, for he was sick.|She smiled, for the news was good.",
        "yet": "It was cold, yet we went out.|He is rich, yet he is unhappy.",
        "so": "It was raining, so we stayed inside.|She was tired, so she went to bed.",
    },
    "Subordinating conjunction": {
        "because": "We stayed in because it was cold.|He smiled because he was happy.",
        "although": "Although it was late, we kept working.|She went out although she felt sick.",
        "though": "Though he was tired, he helped us.|I like it, though it is expensive.",
        "if": "If you call me, I will come.|I do not know if he is home.",
        "when": "Call me when you arrive.|When it rains, the streets get empty.",
        "while": "She read while he cooked.|While I was walking, it started to snow.",
        "since": "I have lived here since 2010.|Since you are here, let us begin.",
        "unless": "We will go unless it rains.|Do not call unless it is urgent.",
        "until": "Wait here until I come back.|He worked until the sun went down.",
        "after": "After we ate, we went for a walk.|She left after the meeting ended.",
        "before": "Wash your hands before you eat.|Before he spoke, the room went quiet.",
        "as": "As she walked in, everyone stood up.|He did as he was told.",
        "whereas": "He likes tea, whereas she prefers coffee.|Some people save, whereas others spend.",
        "once": "Once you try it, you will love it.|Call me once you are home.",
        "that": "I know that she is right.|He said that he would come.",
        "whether": "I wonder whether he will come.|She asked whether the shop was open.",
        "than": "She is taller than her brother.|This is easier than I thought.",
    },
    "Interjection": {
        "oh": "Oh, I forgot my keys!|Oh, what a beautiful day!",
        "wow": "Wow, that is amazing!|Wow, you did it!",
        "ouch": "Ouch, that hurt!|Ouch! I burned my finger.",
        "hey": "Hey, wait for me!|Hey, how are you?",
        "oops": "Oops, I dropped it.|Oops, wrong number!",
        "alas": "Alas, it was too late.|Alas, the story ends here.",
        "hello": "Hello, how are you?|Hello! Is anyone home?",
        "hi": "Hi, nice to meet you.|Hi there!",
        "yes": "Yes, I will help you.|Yes! We won!",
        "no": "No, thank you.|No! Do not touch that.",
        "well": "Well, let us begin.|Well, that was unexpected.",
        "ah": "Ah, now I understand.|Ah, that smells good!",
        "ha": "Ha! I knew it.|Ha, that is funny.",
        "hmm": "Hmm, let me think.|Hmm, I am not sure.",
        "ugh": "Ugh, this is so heavy.|Ugh, not again!",
        "yay": "Yay, we are going to the beach!|Yay! It is Friday.",
        "please": "Please, sit down.|Please, help me.",
        "thanks": "Thanks, I really appreciate it.|Thanks for coming!",
    },
    "Determiner": {
        "a": "She has a cat.|I need a pen.",
        "an": "He ate an apple.|She is an engineer.",
        "the": "The sun is bright.|Close the door, please.",
        "this": "This book is mine.|I like this song.",
        "that": "That house is old.|Look at that bird!",
        "these": "These shoes are new.|I love these flowers.",
        "those": "Those people are friendly.|Look at those clouds.",
        "my": "My brother lives here.|This is my bag.",
        "your": "Is this your phone?|Your idea is great.",
        "his": "He lost his keys.|His voice is calm.",
        "her": "She forgot her umbrella.|Her house is big.",
        "its": "The dog wagged its tail.|The city is known for its parks.",
        "our": "Our train leaves at six.|We love our town.",
        "their": "They sold their car.|Their garden is beautiful.",
        "some": "I bought some bread.|Some people never give up.",
        "any": "Do you have any questions?|I do not have any money.",
        "many": "Many people came to the party.|She has many friends.",
        "much": "I do not have much time.|How much sugar do you want?",
        "few": "Few people know the answer.|I have a few ideas.",
        "every": "Every child needs love.|She walks every morning.",
        "each": "Each student got a book.|Each day is different.",
        "all": "All the lights were on.|All children love games.",
        "no": "There is no milk left.|He has no idea.",
        "both": "Both doors are open.|Both sisters like music.",
        "another": "Would you like another cup?|Let us try another way.",
        "several": "I saw several birds.|Several people were waiting.",
        "more": "I need more time.|Add more salt.",
        "most": "Most students passed.|Most days are quiet.",
        "other": "The other door is locked.|Ask the other teacher.",
        "what": "What time is it?|What book are you reading?",
        "which": "Which road should we take?|Which color do you like?",
        "whose": "Whose bag is this?|Whose turn is it?",
        "either": "You can take either road.|Either answer is fine.",
        "neither": "Neither answer is correct.|Neither player won.",
        "enough": "We have enough food.|Is there enough room?",
        "half": "Half the class was absent.|Half the cake is gone.",
        "such": "It was such a long day.|Such things happen.",
    },
    "Preposition": {
        "in": "The keys are in the drawer.|We met in May.",
        "on": "The book is on the table.|See you on Monday.",
        "at": "Meet me at the station.|The class starts at nine.",
        "to": "I am going to school.|She gave the book to him.",
        "from": "He comes from Spain.|I work from home.",
        "with": "She came with her sister.|Cut it with a knife.",
        "by": "The book was written by her.|We sat by the river.",
        "for": "This gift is for you.|I waited for an hour.",
        "of": "A cup of tea, please.|He is afraid of dogs.",
        "about": "We talked about the trip.|It is a book about space.",
        "over": "The plane flew over the city.|We talked over dinner.",
        "under": "The cat is under the bed.|He is under pressure.",
        "between": "Sit between us.|It is a secret between friends.",
        "through": "We walked through the forest.|She looked through the window.",
        "during": "He slept during the movie.|It rained during the night.",
        "into": "She jumped into the lake.|He turned into a doctor.",
        "onto": "The cat jumped onto the sofa.|He put the box onto the shelf.",
        "across": "He swam across the river.|The shop is across the street.",
        "against": "She leaned against the wall.|We played against a strong team.",
        "behind": "The garden is behind the house.|He hid behind a tree.",
        "beside": "She sat beside me.|The lamp is beside the bed.",
        "near": "I live near the park.|Is there a bank near here?",
        "without": "He left without a word.|Do not go without me.",
        "within": "It is within walking distance.|Reply within two days.",
        "after": "We left after lunch.|After class, I went home.",
        "before": "Come before noon.|She arrived before me.",
        "around": "We walked around the lake.|They sat around the fire.",
        "among": "She was among friends.|He is popular among students.",
        "toward": "He walked toward the door.|She felt kindly toward him.",
        "towards": "The car came towards us.|He ran towards the sea.",
        "until": "Wait until noon.|We stayed until midnight.",
        "up": "He climbed up the hill.|They walked up the stairs.",
        "down": "She ran down the street.|The ball rolled down the hill.",
        "off": "He fell off his bike.|Keep off the grass.",
        "out": "He walked out the door.|She looked out the window.",
        "above": "The sign is above the door.|Birds flew above us.",
        "below": "The temperature is below zero.|Sign your name below the line.",
        "beyond": "The village is beyond the hills.|It was beyond my control.",
        "along": "We walked along the beach.|Trees grow along the road.",
        "upon": "Once upon a time, there was a king.|She relied upon his help.",
        "like": "She sings like an angel.|He looks like his father.",
        "as": "He works as a teacher.|She dressed as a pirate.",
        "since": "I have waited since morning.|She has lived here since 2015.",
        "despite": "We went out despite the rain.|He smiled despite his pain.",
        "inside": "The cat is inside the box.|Stay inside the house.",
        "outside": "The kids are outside the school.|Wait outside the door.",
        "past": "We walked past the church.|It is past midnight.",
        "except": "Everyone came except Tom.|I like all fruit except pears.",
        "per": "The car goes 60 miles per hour.|It costs five dollars per kilo.",
        "than": "He is taller than me.|It costs more than I thought.",
        "unlike": "Unlike her sister, she loves sports.|It is unlike him to be late.",
        "near": "I live near the park.|Is there a bank near here?",
    },
}


def _artigo(w):
    return "an" if w[:1] in "aeiou" else "a"


def _exemplos_modelo(categoria, forma, lema):
    """Simple template sentences for open-class words (nouns, verbs, adjectives, adverbs)."""
    if categoria == "Noun":
        return lema, [
            f"Everyone was talking about the {lema}.",
            f"She wrote a story about the {lema}.",
        ]
    if categoria == "Verb":
        return lema, [
            f"She wants to {lema} today.",
            f"We should {lema} before it is too late.",
        ]
    if categoria == "Adjective":
        return forma, [f"It was very {forma}.", f"The {forma} one is mine."]
    if categoria == "Adverb":
        return forma, [f"He answered {forma}.", f"She did the work {forma}."]
    return forma, []


def exemplos(doc, pos_df, max_palavras=10):
    """For each category, the most frequent words with example sentences that are NOT from the text."""
    cont = Counter(t.text.lower() for t in doc if t.is_alpha)
    lemas_por_palavra = {}
    for t in doc:
        if t.is_alpha and t.pos_ in POS_NOMES:
            lemas_por_palavra.setdefault(t.text.lower(), t.lemma_.lower())
    resultado = {}
    for categoria, grupo in pos_df.groupby("Part of Speech"):
        palavras = sorted(grupo["Word"], key=lambda w: -cont[w])
        itens = []
        for w in palavras:
            fixos = EXEMPLOS_FIXOS.get(categoria, {}).get(w)
            if fixos:
                alvo, frases = w, fixos.split("|")
            else:
                alvo, frases = _exemplos_modelo(categoria, w, lemas_por_palavra.get(w, w))
            marcadas = [
                re.sub(rf"\b{re.escape(alvo)}\b", r"*\g<0>*", f, count=1, flags=re.I)
                for f in frases
            ]
            if marcadas:
                itens.append((w, marcadas))
            if len(itens) >= max_palavras:
                break
        resultado[categoria] = itens
    return resultado


PLAYER_HTML = """
<div style="font-family:sans-serif;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
  <button id="play" style="padding:8px 14px;border-radius:8px;border:1px solid #888;cursor:pointer">🔊 Listen</button>
  <button id="stop" style="padding:8px 14px;border-radius:8px;border:1px solid #888;cursor:pointer">⏹️ Stop</button>
  <select id="rate" style="padding:8px;border-radius:8px;border:1px solid #888">
    <option value="0.75">0.75x</option>
    <option value="1" selected>1x</option>
    <option value="1.25">1.25x</option>
  </select>
</div>
<script>
  const text = _TEXT_;
  const chunks = text.match(/[^.!?\\n]+[.!?]*/g) || [text];
  document.getElementById("play").onclick = () => {
    speechSynthesis.cancel();
    const rate = parseFloat(document.getElementById("rate").value);
    chunks.forEach(c => {
      if (!c.trim()) return;
      const u = new SpeechSynthesisUtterance(c.trim());
      u.lang = "en-US";
      u.rate = rate;
      speechSynthesis.speak(u);
    });
  };
  document.getElementById("stop").onclick = () => speechSynthesis.cancel();
</script>
"""


def player_audio(texto):
    seguro = json.dumps(texto).replace("</", "<\\/")
    components.html(PLAYER_HTML.replace("_TEXT_", seguro), height=60)


TEXTOS_EXEMPLO = [
    'Anna woke up early because the storm was loud. The wind shook the windows, and the rain fell hard on the roof of her small house. She looked through the glass, but the street was empty. Oh, the old bridge had fallen into the river! Anna grabbed her yellow coat and ran quickly toward the village. On the way, she met her neighbor, Mr. Brown, who was carrying a heavy lamp. "Where are you going?" he asked. "The bridge is broken," she answered, "and the children cannot reach their school." Together they walked along the muddy road until they arrived at the square. Many people were already there, and they were talking loudly about the problem. The mayor, a tall and tired man, raised his hand. "We need strong ropes, wooden boards and brave volunteers," he said. Anna felt afraid, yet she stepped forward first. Soon, dozens of villagers carried tools and planks to the river. They worked all morning while the rain slowly stopped. By noon, a simple new bridge crossed the water. The children walked over it, laughing and shouting with joy. Anna smiled because she knew the village was stronger than any storm. That evening, everyone shared hot soup in the old church, and nobody felt alone. Wow, what a day it had been!',
    'The little fox lived in a quiet forest near a wide blue lake. Every night, it crept slowly between the tall trees and searched for food. The forest was full of strange sounds, but the fox was clever and careful. Although it was afraid of the big owl, it never gave up. One cold evening, the snow covered the ground, and the fox could not find any berries or mice. Its stomach was empty, so it walked farther than usual. Wow, the moon was bright, and the silver light showed a small farm behind the hills! In the garden, a sweet red apple hung from a low branch. The fox jumped high, but it missed the fruit twice. Then it saw an old wooden box under the tree. It pushed the box carefully with its nose, climbed on top, and finally caught the apple. Suddenly, a loud dog began to bark inside the farmhouse. The fox ran quickly across the field and disappeared into the dark forest. When it reached its home, its heart was beating fast. It ate the apple slowly while the stars shone above the trees. That night, the fox learned an important lesson: patience and courage can solve difficult problems. In the morning, it slept peacefully, and the forest was silent again.',
    'Tom opened the heavy door and entered the dark room. A cold wind moved across the floor, and the candles on the table trembled. He had come to the old house because his grandfather had left him a mysterious letter. The letter said that a hidden key waited inside the library. Tom walked slowly past the dusty shelves and the broken chairs. Hey, who is there? he whispered. Nobody answered, yet he heard a soft sound behind the old mirror. His hands were shaking, but he moved closer. Behind the mirror, he found a narrow staircase that led to a secret cellar. Tom took a deep breath and went down the stairs. At the bottom, a small iron box rested on a wooden table. He lifted the lid carefully. Inside, there was a golden key and a photograph of his grandfather as a young sailor. On the back of the photograph, a short message appeared: "Courage opens every door." Tom smiled, although tears filled his eyes. He put the key in his pocket and climbed back to the library. Outside, the sun was rising, and the garden looked warm and bright. Tom closed the door behind him, and for the first time that year, he felt truly happy and brave.',
    'Maria loves the sea. On Sunday, she and her brother walked along the beach while the sun was going down. The sand was soft and warm under their feet, and the waves made a gentle music. They collected small shells and talked about their new house in the city. Maria was excited, but her brother felt sad because he would miss his friends. "Do not worry," she told him, "we can visit the sea every summer." He nodded slowly and looked at the horizon. Suddenly, a white bird landed beside them and stared at the basket full of shells. Maria laughed, and her brother gave the bird a piece of bread. After that, they sat on a large rock and watched the sky change from orange to pink and then to deep purple. A fisherman passed by in a tiny boat and waved at them kindly. Ah, how peaceful everything was! When the first stars appeared, their mother called them from the road. They picked up the basket and ran home together. That night, Maria put the biggest shell on her desk. She knew that wherever she lived, the sea would always stay in her heart. It was a calm and beautiful evening, and nobody forgot it.',
    'The king was proud, but his people were hungry. The harvest had failed, and the winter was long and cruel. In the royal palace, the tables were full of meat, fruit and sweet cakes, while outside, many families had nothing. A young girl named Lucy lived in a small house near the market. Every morning, she baked bread with the little flour that she had, and she carried it to the poor families in the city. Some people called her foolish because she gave away her only food. However, Lucy always answered with a smile. "A shared piece of bread is better than a full table alone," she said. One cold day, a guard saw her at the palace gate and took her to the king. The king looked at her angrily and asked why she was there. Lucy opened her basket and offered him a warm loaf. The room became silent. When the king heard her story, he felt ashamed. He stood up, called his servants, and ordered them to open the royal gates and the storehouses. Soon, bread and soup reached every street. The people cheered loudly, and the king thanked the brave girl. From that day, he ruled with kindness, and Lucy\'s bakery became the most beloved place in the kingdom.',
]


def mostrar_texto(texto):
    """Shows the text in a box whose height always fits the text."""
    escapado = re.sub(r"([\\`*_{}\[\]()#+\-!>~|$])", r"\\\1", texto)
    paragrafos = [p.strip() for p in escapado.splitlines() if p.strip()]
    with st.container(border=True):
        st.markdown("\n\n".join(paragrafos))


def gerar_texto():
    """Maximum length: all the stories, in a new random order each time."""
    st.session_state["texto_gerado"] = "\n\n".join(
        random.sample(TEXTOS_EXEMPLO, k=len(TEXTOS_EXEMPLO))
    )


# ---------------- Interface ----------------
st.title("📖 Part of Speech")
st.caption("Shows the part of speech of each word in an English text and lets you listen to it.")

modo = st.radio("Choose an option", ["Upload", "Generate text"], horizontal=True)

if modo == "Upload":
    arquivo = st.file_uploader("Upload a .txt file", type=["txt"])
    texto = arquivo.read().decode("utf-8", errors="ignore") if arquivo else ""
else:
    st.button("🎲 Generate text", on_click=gerar_texto)
    texto = st.session_state.get("texto_gerado", "")

if texto:
    mostrar_texto(texto)

if st.button("Analyze", type="primary"):
    if not texto.strip():
        st.warning("Upload a file or generate a text first.")
        st.stop()

    with st.spinner("Analyzing..."):
        nlp = carregar_modelo()
        doc = nlp(texto)
        pos_df = classes_gramaticais(doc)
        exem = exemplos(doc, pos_df)

    player_audio(texto)

    if pos_df.empty:
        st.info("No words found.")
    else:
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
        for categoria in dict.fromkeys(POS_NOMES.values()):
            if categoria in set(pos_df["Part of Speech"]):
                with st.expander(f"How to use: {categoria}"):
                    st.markdown(COMO_USAR[categoria])
                    if exem.get(categoria):
                        st.markdown("*More examples:*")
                        for palavra, frases in exem[categoria]:
                            st.markdown(
                                f"*{palavra}*\n" + "\n".join(f"- {f}" for f in frases)
                            )