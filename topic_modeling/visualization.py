import itertools
import numpy as np
import pandas as pd
from typing import List, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pyLDAvis
from modules.config.constants import log_dir

# Barchart for top 5 keyword weights by topic
def visualize_topic_term(
    topic_model,
    topics: List[int] = None,
    top_n_topics: int = None,
    n_words: int = 10,
    custom_labels: Union[bool, str] = True,
    title: str = "<b>Top 5 Keywords per Topic</b>",
    width: int = 400,
    height: int = 400,
    autoscale: bool = False,
) -> go.Figure:

    colors = itertools.cycle(["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"])

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Initialize figure
    if isinstance(custom_labels, str):
        subplot_titles = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        subplot_titles = ["_".join([label[0] for label in labels[:4]]) for labels in subplot_titles]
        subplot_titles = [label if len(label) < 30 else label[:27] + "..." for label in subplot_titles]
    elif topic_model.custom_labels_ is not None and custom_labels:
        subplot_titles = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topics]
    else:
        subplot_titles = [f"Topic {topic}" for topic in topics]
    columns = 3
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(
        rows=rows,
        cols=columns,
        shared_xaxes=False,
        horizontal_spacing=0.1,
        vertical_spacing=0.2 / rows if rows > 1 else 0.1,
        subplot_titles=subplot_titles,
    )

    # Add barchart for each topic
    row = 1
    column = 1
    for topic in topics:
        words = [word + "  " for word, _ in topic_model.get_topic(topic)][:n_words][::-1]
        scores = [score for _, score in topic_model.get_topic(topic)][:n_words][::-1]

        fig.add_trace(
            go.Bar(x=scores, y=words, orientation="h", marker_color=next(colors)),
            row=row,
            col=column,
        )

        if autoscale:
            if len(words) > 12:
                height = 250 + (len(words) - 12) * 11

            if len(words) > 9:
                fig.update_yaxes(tickfont=dict(size=(height - 140) // len(words)))

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            "text": f"{title}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        width=width * 4,
        height=height * rows if rows > 1 else height * 1.5,
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Source Sans Pro"),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    pio.write_html(fig, file=f'{log_dir}/topic_term_viz.html', auto_open=False)
    

# Visualize topic hierarchy
def visualize_topic_hierarchy(topic_model, docs):
    from scipy.cluster import hierarchy as sch

    linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

    # Extract hierarchical topics and their representations.
    # A dataframe that contains a hierarchy of topics represented by their parents and their children.
    hierarchical_topics: pd.DataFrame = topic_model.hierarchical_topics(docs, linkage_function=linkage_function)

    fig = topic_model.visualize_hierarchy(
        orientation='left',
        hierarchical_topics=hierarchical_topics,
        custom_labels=True,
        width=1200,
        height=1000,
    )

    fig.update_layout(
        # Adjust left, right, top, bottom margin of the overall figure.
        margin=dict(l=20, r=20, t=60, b=20),

        title={
            'text': "Hierarchical structure of the topics",
            'y':0.975,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="#000000"
            )
        },
    )

    pio.write_html(fig, file=f'{log_dir}/topic_hierarchy_viz.html', auto_open=False)

# Visualize topic clusters in pyLDAvis
def visualize_topic_clusters(topic_model, probs, docs):
    import numpy as np
    topic_term_dists = topic_model.c_tf_idf_.toarray()
    outlier = np.array(1 - probs.sum(axis=1)).reshape(-1, 1)
    doc_topic_dists = np.hstack((probs, outlier))

    doc_lengths = [len(doc) for doc in docs]
    vocab = [word for word in topic_model.vectorizer_model.vocabulary_.keys()]
    term_frequency = [topic_model.vectorizer_model.vocabulary_[word] for word in vocab]

    data = {'topic_term_dists': topic_term_dists,
            'doc_topic_dists': doc_topic_dists,
            'doc_lengths': doc_lengths,
            'vocab': vocab,
            'term_frequency': term_frequency}

    viz = pyLDAvis.prepare(**data,sort_topics=False, n_jobs = 1, mds='mmds')
    pyLDAvis.save_html(viz, f'{log_dir}/pyLDAvis.html')