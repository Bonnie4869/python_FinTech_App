import csv
from fileinput import filename

import requests
from lxml import etree
url = 'https://oir.centanet.com/lease/?pageindex={}'
headers = {
    'user-agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 '
                 '(KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36 Edg/135.0.0.0'
}
f = open('store.csv',mode='w', newline='', encoding='utf-8')
filenames = ['name','floor','usage','area','avg_price_monthly','transportation_time', 'district','year','air_con']
write = csv.DictWriter(f, fieldnames = filenames)
write.writeheader()
for i in range(2, 577):
    response = requests.post(url=url.format(i), headers=headers).text
    tree = etree.HTML(response)
    div_list = tree.xpath('//div[@class="property-results-content"]/div')
    sale_office_urls = []
    all_office_urls = []
    for div in div_list:
        item = {}
        name_list = div.xpath('.//div[@class="property-name"]/span[1]/text()')
        item['name'] = name_list[0] if name_list else None
        floor_list = div.xpath('.//div[@class="property-name"]/span[2]/text()')
        item['floor'] = floor_list[0] if floor_list else None
        usage_list = div.xpath('.//p[@class="usage-size"]/span[1]/text()')
        item['usage'] = usage_list[0] if usage_list else None
        area_list = div.xpath('.//p[@class="usage-size"]/span[3]/text()')
        item['area'] = area_list[0] if area_list else None
        avg_price_monthly_list = div.xpath('.//span[contains(@class, "avgPrice")]/text()')
        item['avg_price_monthly'] = [price.replace('@', '') for price in avg_price_monthly_list] if avg_price_monthly_list else None
        transportation_time_list = div.xpath('.//div[@class="tag-list"]/span[1]/text()')
        item['transportation_time'] = transportation_time_list[0] if transportation_time_list else None
        half_url_list = div.xpath('.//a[@class="property-results-card"]/@href')
        half_url = half_url_list[0] if half_url_list else None
        if half_url:
            detail_url = 'https://oir.centanet.com' + half_url
            if half_url.startswith("/lease/office/"):
                sale_office_urls.append(detail_url)
            elif half_url.startswith("/all/office/"):
                all_office_urls.append(detail_url)
        detail_response = requests.post(url=detail_url, headers=headers).text
        detail_tree = etree.HTML(detail_response)
        item['district'] = detail_tree.xpath('//div[@class="breadcrumb"]//a[1]/span/text()')
        year_list = detail_tree.xpath('//div[@class="text_1"][5]/p/text()')
        item['year'] = year_list[0] if year_list else None
        air_con_list = detail_tree.xpath('//div[@class="text_1"][6]/p/text()')
        item['air_con'] = air_con_list[0] if air_con_list else None
        write.writerow(item)


