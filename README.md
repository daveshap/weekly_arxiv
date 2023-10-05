# weekly_arxiv

- Main Export Link: https://export.arxiv.org/api/query?search_query=all:llms&sortBy=lastUpdatedDate&sortOrder=descending&max_results=200 

## Usage

- run `python download_latest.py`
- it will update `output.md`

You can modify the link in the script to change the search term, sorting, and limits in the URL


## Claud Prompting

Claude has 100k token window, which makes it useful for surveying the resulting documents

### Identifying trends

Copy and paste the Markdown into Claude and then use this to get a list of trends

```text
Can you please characterize the major trends in the latest LLM research. ONLY use the material I have given you here. The goal is to summarize the last week of LLM research.
```

### Listing out papers for specific trends

Once you identify trends, you can use this example. In this case I was asking about benchmarks.

```text
Please write a summary of the benchmarking trends as elucidated by the text I gave you. The summary should be in the format of a complete paragraph followed by a list of key innovations. Each item in the list should include the paper link and paper title, along with a brief description of the innovation. Make sure you only pull from the information I gave you in this conversation.
```