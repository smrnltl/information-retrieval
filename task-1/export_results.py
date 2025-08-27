#!/usr/bin/env python3
"""
Export final results from database to JSON
"""

import sqlite3
import json

def export_to_json():
    print("Exporting publications from database to JSON...")
    
    # Connect to database
    conn = sqlite3.connect('publications.db')
    c = conn.cursor()
    
    # Get all publications
    c.execute('SELECT title, authors, abstract, year, link FROM publications ORDER BY year DESC, title ASC')
    rows = c.fetchall()
    conn.close()
    
    publications_data = []
    successful_extractions = 0
    
    for row in rows:
        try:
            authors_array = json.loads(row[1]) if row[1] else []
        except:
            authors_array = []
        
        publication = {
            'title': row[0] or '',
            'authors': authors_array,
            'abstract': row[2] or '',
            'year': row[3] or '',
            'link': row[4] or ''
        }
        
        publications_data.append(publication)
        
        # Count successful extractions (those with authors and abstracts)
        if authors_array and row[2]:
            successful_extractions += 1
    
    # Export to JSON
    with open('publications_final.json', 'w', encoding='utf-8') as f:
        json.dump(publications_data, f, ensure_ascii=False, indent=2)
    
    print(f"SUCCESS: Exported {len(publications_data)} publications to publications_final.json")
    print(f"Publications with authors and abstracts: {successful_extractions}")
    print(f"Success rate: {successful_extractions/len(publications_data)*100:.1f}%" if publications_data else "0%")
    
    # Show year distribution
    year_counts = {}
    for pub in publications_data:
        year = pub['year'] or 'Unknown'
        year_counts[year] = year_counts.get(year, 0) + 1
    
    print("\nYear distribution:")
    for year in sorted(year_counts.keys(), reverse=True):
        print(f"  {year}: {year_counts[year]} publications")

if __name__ == "__main__":
    export_to_json()