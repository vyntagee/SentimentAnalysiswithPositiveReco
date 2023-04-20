from flask import Flask, render_template, request, jsonify
import plotly.express as px
import plotly.io as pio
import pandas as pd
from thesis import analyze_tweets

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_query = request.form['search_query']
        results = analyze_tweets(search_query)

        # Create pie charts for 'category' and 'sentiment'
        data_sorted = pd.DataFrame(results)
        category_pie_chart = px.pie(data_sorted, names='category', title='Category Distribution')
        sentiment_pie_chart = px.pie(data_sorted, names='sentiment', title='Sentiment Distribution')

        # Convert pie charts to HTML
        category_pie_chart_html = pio.to_html(category_pie_chart, full_html=False)
        sentiment_pie_chart_html = pio.to_html(sentiment_pie_chart, full_html=False)

        return render_template('results.html', results=results, category_pie_chart=category_pie_chart_html, sentiment_pie_chart=sentiment_pie_chart_html)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
