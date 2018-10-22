---
title: "Scraping Dance-Off Data from Wikipedia"
date: 2018-10-21T18:48:40-04:00
draft: false
---

Now that I have [two](../score-evaluation/) [weeks](../week-5-scores) of predicting Strictly scores under my belt, I'm interested in looking at who gets voted into the dance-off. For example, [a 2016 article](https://www.theguardian.com/commentisfree/2016/dec/16/black-strictly-come-dancing) in *The Guardian* by Julia Carter and Richard McManus reported that

>  "after controlling for where the couple have come in the judges’ scoring, an ethnic minority celebrity is statistically significantly more likely to be in the bottom two and therefore to have received a lower public vote."

They say their study found a 71% increase in the likelihood of being in the dance-off for non-white celebrities, and 83% for black women—strikingly large effects. It doesn't seem like a more detailed description of their analysis has been published, so I am interested in whether I could reproduce their results.

It would also be interesting to look at other possible predictive variables: e.g., size of social media following, age, hometown, or type of celebrity.

Because the BBC has never made information about the public voting results public, the only outcome that can be assessed is the identity of the two contestants in the dance-off each week.

The first step is gathering the data. Unfortunately, [Ultimate Strictly](http://www.ultimatestrictly.com/) doesn't provide a full set of dance-off data. Instead, I scraped the data from the voluminous Strictly series-specific Wikipedia pages.

The Wikipedia pages do provide dance-off data for all series, but dance-off information is only summarized a full series at a time by the background color of cells in each series' "Scoring chart" (e.g., [for Series 15](https://en.wikipedia.org/wiki/Strictly_Come_Dancing_(series_15)#Scoring_chart)).

Despite this inconvenience, the structure of the Wikipedia pages are uniform enough that I was able to construct a Python script to scrape the couples in each week's dance-off. As an overview, the steps were to:

- Use `requests` to download the html.
- Use Beautiful Soup 4 (`bs4`) to navigate the parse tree of the html and isolate the "Scoring chart" table.
- Use and slightly modify a [very helpful function I found on StackOverflow](https://stackoverflow.com/a/48451104/4280216) that converts an html table that uses `rowspan` (i.e., a single cell stretching over multiple rows) into a nested list representing a standard *m x n* array.
- Write a function, `check_danceoff()`, that determines whether a table cell has the orange (eliminated) or light blue (survived the dance-off) color that denotes being in the dance-off.
- Pipe everything together in a function `find_danceoffs()` that returns a tidy DataFrame in which each row corresponds to an observation of a single partnership in the dance-off for a single week.

I'll append all the scraping code to the bottom of this post, but first a look at the final collection of the entire dance-off data set:

```python
df_all = pd.DataFrame(columns=['celebrity','professional','week','series'])
for x in range(16):
    series_str = 'series_{}'.format(x+1)
    scd_url = 'https://en.wikipedia.org/wiki/Strictly_Come_Dancing_({})'.format(series_str)
    df_series = find_danceoffs(scd_url)
    df_series['series'] = x+1
    df_all = df_all.append(df_series)
```

And then `df_all.tail()` gives (Wikipedia was incredibly fast to update with today's series 16 results!):

```
|     | celebrity | professional | week | series |
|-----|-----------|--------------|------|--------|
| 309 | Lee       | Nadiya       | 3    | 16     |
| 310 | Charles   | Karen        | 4    | 16     |
| 311 | Katie     | Gorka        | 4    | 16     |
| 312 | Seann     | Katya        | 5    | 16     |
| 313 | Vick      | Graziano     | 5    | 16     |
```

And `df_describe()`:

```
|        | celebrity | professional | week | series |
|--------|-----------|--------------|------|--------|
| count  | 314       | 314          | 314  | 314    |
| unique | 170       | 49           | 13   | 16     |
| top    | Mark      | Brendan      | 5    | 7      |
| freq   | 10        | 28           | 32   | 24     |
```

Turns out Brendan is the professional that's been in the dance-off more than anyone else.

{{< figure src="/strictly-come-data/images/bruno-no.gif">}}

Rest of the code pasted below. And remember, keeeeeeeeeep data-ing!

```python
import bs4
import numpy as np
import pandas as pd
import requests

def check_danceoff(ele):
    if ele.has_attr('style'):
        # remove whitespace and ;
        style = ele.attrs['style'].replace(' ','').replace(';','')
        if style == 'background:lightblue':
            return True
        elif style == 'background:orange':
            return True
        else:
            return False
    elif ele.has_attr('bgcolor'):
        bgcolor = ele.attrs['bgcolor']
        if (bgcolor=='orange') or (bgcolor=='lightblue'):
            return True
        else:
            return False
    else:
        return False

# from https://stackoverflow.com/a/48451104/4280216
# use because of rowspans
# this function handles rowspans to return array with
# equal number of entries for each row
#
# I modified so the value returned for each cell is
# cell_func(cell) rather than cell.get_text().
def table_to_2d(table_tag, cell_func):
    rowspans = []  # track pending rowspans
    rows = table_tag.find_all('tr')

    # first scan, see how many columns we need
    colcount = 0
    for r, row in enumerate(rows):
        cells = row.find_all(['td', 'th'], recursive=False)
        # count columns (including spanned).
        # add active rowspans from preceding rows
        # we *ignore* the colspan value on the last cell, to prevent
        # creating 'phantom' columns with no actual cells, only extended
        # colspans. This is achieved by hardcoding the last cell width as 1. 
        # a colspan of 0 means “fill until the end” but can really only apply
        # to the last cell; ignore it elsewhere. 
        colcount = max(
            colcount,
            sum(int(c.get('colspan', 1)) or 1 for c in cells[:-1]) + len(cells[-1:]) + len(rowspans))
        # update rowspan bookkeeping; 0 is a span to the bottom. 
        rowspans += [int(c.get('rowspan', 1)) or len(rows) - r for c in cells]
        rowspans = [s - 1 for s in rowspans if s > 1]

    # it doesn't matter if there are still rowspan numbers 'active'; no extra
    # rows to show in the table means the larger than 1 rowspan numbers in the
    # last table row are ignored.

    # build an empty matrix for all possible cells
    table = [[None] * colcount for row in rows]

    # fill matrix from row data
    rowspans = {}  # track pending rowspans, column number mapping to count
    for row, row_elem in enumerate(rows):
        span_offset = 0  # how many columns are skipped due to row and colspans 
        for col, cell in enumerate(row_elem.find_all(['td', 'th'], recursive=False)):
            # adjust for preceding row and colspans
            col += span_offset
            while rowspans.get(col, 0):
                span_offset += 1
                col += 1

            # fill table data
            rowspan = rowspans[col] = int(cell.get('rowspan', 1)) or len(rows) - row
            colspan = int(cell.get('colspan', 1)) or colcount - col
            # next column is offset by the colspan
            span_offset += colspan - 1
            # define actual value to put in output table
            value = cell_func(cell)
            for drow, dcol in product(range(rowspan), range(colspan)):
                try:
                    table[row + drow][col + dcol] = value
                except IndexError:
                    # rowspan or colspan outside the confines of the table
                    pass

        # update rowspan bookkeeping
        rowspans = {c: s - 1 for c, s in rowspans.items() if s > 1}

    return table

def find_danceoffs(url):
    r = requests.get(url)
    soup = bs4.BeautifulSoup(r.text, 'html.parser')
    try:
        chart_span = soup.find('span',attrs={'id':'Scoring_chart'})
        score_table = chart_span.findNext(name='table')
        table_body = score_table.find('tbody')

        text_array = np.array(table_to_2d(table_body, lambda c: c.get_text().rstrip()))
        for_pd = {x[0]: x[1:] for x in text_array.T}
        df = pd.DataFrame(for_pd)
        # return df columns to original order and set 'Couple' as index
        if 'Couple' in df.columns:
            df = df[text_array[0]].set_index("Couple")
        # this handles inconsistent column name in Series 5 page
        elif 'Team' in df.columns:
            df = df[text_array[0]].set_index("Team")
            df.index.name = "Couple"
        else:
            raise Exception

        danceoff_mask = np.array(table_to_2d(table_body, check_danceoff))
        # convert to DataFrame with same columns and index as df
        danceoff_df = pd.DataFrame(danceoff_mask[1:,1:], columns=df.columns, index=df.index)

        df_filter = df[danceoff_df]
        danceoffs = {}
        
        episode_list = []
        celeb_list = []
        pro_list = []
        for col in df_filter.columns:
            episode = df_filter[col].dropna().reset_index()
            if len(episode)!=0:
                # clean up episode name for output dict:
                if col=='1+2':
                    clean_name = 2
                else:
                    only_digits = ''.join(c for c in col if c.isdigit())
                    clean_name = int(only_digits)
                
                # split list of couples into celeb and pro first names
                celebs = episode.apply(lambda x: x.Couple.split('&')[0].strip(), axis=1)
                pros = episode.apply(lambda x: x.Couple.split('&')[1].strip(), axis=1)
                for celeb, pro in zip(celebs,pros):
                    episode_list.append(clean_name)
                    celeb_list.append(celeb)
                    pro_list.append(pro)

    except Exception as e:
        print('problem with {}: {}'.format(url, e))

    danceoffs_dict = {'week': episode_list, 'celebrity': celeb_list, 'professional': pro_list}
    danceoffs = pd.DataFrame(danceoffs_dict)
    return danceoffs
```
