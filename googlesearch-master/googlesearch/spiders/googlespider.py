from urllib.parse import urljoin, urlparse, parse_qsl
import datetime
from scrapy.http import Request
from scrapy.selector import HtmlXPathSelector
from scrapy.spider import BaseSpider
from scrapy.utils.response import get_base_url
from scrapy.utils.misc import arg_to_iter
from googlesearch.items import GoogleSearchItem

COUNTRIES = {
    'ie': 'countryIE',
    'nl': 'countryNL'
}

"""
A spider to parse the google search result bootstraped from given queries.
"""
class GoogleSearchSpider(BaseSpider):
    name = 'googlesearch'
    queries = ('contact us', 'hotel')
    region = 'ie'
    download_delay = 1
    base_url_fmt = 'https://www.google.com/search?q={query}&rlz=1C1CHZL_enSG815SG815&oq=women+floral+lace+long+sleeve+off+shoulder+wedding+mermaid+cocktail+dress&aqs=chrome..69i57.6318j0j7&sourceid=chrome&ie=UTF-8'
    download_html = False
    limit_country = False

    def start_requests(self):
        for query in arg_to_iter(self.queries):
            url = self.make_google_search_request(COUNTRIES[self.region], query)
            yield Request(url=url, meta={'query': query})

    def make_google_search_request(self, country, query):
        if not self.limit_country:
            country = ''
        return self.base_url_fmt.format(country=country, region=self.region, query='+'.join(query.split()).strip('+'))

    def parse(self, response):
        hxs = HtmlXPathSelector(response)
        SET_SELECTOR = '.g'
        for result in response.css(SET_SELECTOR):
            NAME_SELECTOR = 'h3 a::text'
            print(result.css(NAME_SELECTOR).extract_first())
        next_page = hxs.select('//table[@id="nav"]//td[contains(@class, "b") and position() = last()]/a')
        if next_page:
            url = self._build_absolute_url(response, next_page.select('.//@href').extract()[0])
            print("next")
            yield Request(url=url, callback=self.parse, meta={'query': response.meta['query']})
#        hxs = HtmlXPathSelector(response)
#        print(len(hxs.select('//div[@id="srg"]//div[@class="g"]')))
#        for sel in hxs.select('//div[@id="srg"]//div[@class="g"]'):
#            name = u''.join(sel.select(".//text()").extract())
#            url = _parse_url(sel.select('.//a/@href').extract()[0])
#            region = _get_region(url)
#            print("\n"+name+"\n")
#            if len(url):
#                if self.download_html:
#                    yield Request(url=url, callback=self.parse_item, meta={'name':name,
#                                                                           'query': response.meta['query']})
#                else:
#                    yield GoogleSearchItem(url=url, name=name, query=response.meta['query'], crawled=datetime.datetime.utcnow().isoformat())
#
#        next_page = hxs.select('//table[@id="nav"]//td[contains(@class, "b") and position() = last()]/a')
#        if next_page:
#            url = self._build_absolute_url(response, next_page.select('.//@href').extract()[0])
#            print("next")
#            yield Request(url=url, callback=self.parse, meta={'query': response.meta['query']})

    def parse_item(self, response):
        name = response.meta['name']
        query = response.meta['query']
        url = response.url
        html = response.body[:1024 * 256]
        timestamp = datetime.datetime.utcnow().isoformat()
        print("\n"+name+"\n")
        yield GoogleSearchItem({'name': name,
                                'url': url,
                                'html': html,
                                'region': self.region,
                                'query': query,
                                'crawled': timestamp})

    def _build_absolute_url(self, response, url):
        return urljoin(get_base_url(response), url)

def _parse_url(href):
    """
    parse the website from anchor href.

    for example:

    >>> _parse_url(u'/url?q=http://www.getmore.com.hk/page18.php&sa=U&ei=Xmd_UdqBEtGy4AO254GIDg&ved=0CDQQFjAGODw&usg=AFQjCNH08dgfL10dJVCyQjfu_1DEyhiMHQ')
    u'http://www.getmore.com.hk/page18.php'
    """
    queries = dict(parse_qsl(urlparse(href).query))
    return queries.get('q', '')

def _get_region(url):
    """
    get country code from the url.

    >>> _get_region('http://scrapinghub.ie')
    'ie'
    >>> _get_region('http://www.astoncarpets.ie/contact.htm')
    'ie'
    """
    netloc = urlparse(url)[1]
    return netloc.rpartition('.')[-1]
