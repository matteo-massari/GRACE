from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.switches = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and attrs.get('href'):
            self.links.append(attrs['href'])
            if attrs.get('hreflang') in {'it', 'en'}:
                self.switches.append(attrs['href'])

root = Path('docs/_build/html')
errors = []
pages = list(root.rglob('*.html'))
for p in pages:
    parser = Links()
    parser.feed(p.read_text())
    if len(parser.switches) != 1:
        errors.append(f'{p}: expected one language switch, got {parser.switches}')
    for href in parser.links:
        url = urlsplit(href)
        if url.scheme or url.netloc or not url.path:
            continue
        target = p.parent / unquote(url.path)
        if not target.exists():
            errors.append(f'{p}: missing {href}')
    for href in parser.switches:
        target = (p.parent / href).resolve()
        rel = p.relative_to(root).as_posix()
        if '_modules/' not in rel:
            expected = root / (rel[3:] if rel.startswith('it/') else 'it/' + rel)
            if target != expected.resolve():
                errors.append(f'{p}: switch points to wrong page: {href}')
assert not errors, '\n'.join(errors)
print(f'Validated local links and matching-page language switches in {len(pages)} HTML pages.')
