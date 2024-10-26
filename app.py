from flask import Flask, request, render_template, session
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, random
from dicts import sentiment_labels, motivational_responses, apology_responses, neutral_responses

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Завантажуємо модель та токенізатор DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Налаштування для аналізу сентименту з новою моделлю
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Головна сторінка
@app.route('/')
def home():
    session['messages'] = []
    session['chat_history_ids'] = None
    return render_template('index.html', messages=session['messages'])


# Обробка запитів
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if session.get('chat_history_ids') is not None:
        chat_history_ids = torch.cat([torch.tensor(session['chat_history_ids']), new_user_input_ids], dim=-1)
    else:
        chat_history_ids = new_user_input_ids

    bot_output = model.generate(chat_history_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(bot_output[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

    # Аналіз сентименту
    sentiments = sentiment_analyzer(response_text)
    sentiment_label = sentiments[0]['label']
    sentiment = sentiment_labels[sentiment_label]
    # Перетворення числового сентименту на текст
    sentiment_output = f"Сентимент: {sentiment}"


    # Генерація нової відповіді в залежності від сентименту
    if sentiment == "positive":  # позитивний
        follow_up_response = random.choice(motivational_responses)
    elif sentiment == "negative":  # негативний
        follow_up_response = random.choice(apology_responses)
    else:  # нейтральний
        follow_up_response = random.choice(neutral_responses)

    # Збереження історії діалогу
    session['chat_history_ids'] = chat_history_ids.tolist()
    session['messages'].append({'role': 'user', 'content': user_input})
    session['messages'].append({'role': 'bot', 'content': response_text + ". " + follow_up_response})
    session['messages'].append({'role': 'sentiment', 'content': sentiment_output})

    return render_template('index.html', messages=session['messages'])


# Очистка історії чату
@app.route('/reset', methods=['POST'])
def reset():
    session.pop('messages', None)
    session.pop('chat_history_ids', None)
    return render_template('index.html', messages=[])


if __name__ == '__main__':
    app.run(debug=True)