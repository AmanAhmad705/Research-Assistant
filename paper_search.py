import requests

def search_papers(query, limit=10):
    """
    Search for research papers using Semantic Scholar API.
    Returns a list of papers with title, abstract, PDF URL, etc.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,authors,citationCount,openAccessPdf"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json().get("data", [])

        # Filter and format
        papers = []
        for item in data:
            pdf_url = item.get("openAccessPdf", {}).get("url")
            papers.append({
                "title": item.get("title"),
                "abstract": item.get("abstract"),
                "year": item.get("year"),
                "authors": [a["name"] for a in item.get("authors", [])],
                "citationCount": item.get("citationCount"),
                "pdf_url": pdf_url
            })

        return papers

    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        return []
