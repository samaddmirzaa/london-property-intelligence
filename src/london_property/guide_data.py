"""Download the GOV.UK guides the chatbot can quote, one row per guide part, for the get_guide tool.

Run from the project root (needs internet):
    python -m london_property.guide_data
"""
import json
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup

from london_property.tools import area_dir, guide_slugs

api = 'https://www.gov.uk/api/content/'


def html_to_text(html):
    """GOV.UK page HTML as plain text: one line per paragraph or list item, table rows as 'cell | cell'."""
    soup = BeautifulSoup(html, 'html.parser')
    for row in soup.find_all('tr'):
        row.replace_with(' | '.join(cell.get_text(' ', strip=True) for cell in row.find_all(['th', 'td'])) + '\n')
    for block in soup.find_all(['h2', 'h3', 'h4', 'p', 'li']):
        block.append('\n')
    lines = (' '.join(line.split()) for line in soup.get_text().splitlines())
    return '\n'.join(line for line in lines if line)


def fetch_guide(slug):
    """One GOV.UK guide from the Content API, as one row per part."""
    page = json.load(urlopen(api + slug, timeout=30))
    return [{'guide': slug, 'guide_title': page['title'], 'part': part['slug'], 'part_title': part['title'].strip(),
             'url': f"https://www.gov.uk/{slug}/{part['slug']}", 'updated': page['public_updated_at'][:10],
             'text': html_to_text(part['body'])}
            for part in page['details']['parts']]


def main():
    table = pd.DataFrame([row for slug in guide_slugs for row in fetch_guide(slug)])
    table.to_parquet(area_dir/'guides.parquet', index=False)
    print(f'guides: {table.shape}')


if __name__ == '__main__':
    main()