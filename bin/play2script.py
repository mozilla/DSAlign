import sys
import json
from xml.dom import minidom


def fail():
    print('Usage: play2script.py (script|plain|lines) <play-file.xml> <output>')
    exit(1)


def get_text(elements):
    return ' '.join(map(lambda element: ' '.join(t.nodeValue.strip() for t in
                                                 element.childNodes if t.nodeType == t.TEXT_NODE),
               elements))


def main(args):
    if len(args) != 3:
        fail()
    dom = minidom.parse(args[1])
    if args[0] == 'script':
        script = []
        for speech in dom.getElementsByTagName('SPEECH'):
            speaker = get_text(speech.getElementsByTagName('SPEAKER'))
            speaker = ' '.join(map(lambda p: p[0] + p[1:].lower(), speaker.split(' ')))
            text = get_text(speech.getElementsByTagName('LINE'))
            script.append({
                'speaker': speaker,
                'text': text
            })
        with open(args[2], 'w') as script_file:
            script_file.write(json.dumps(script))
    elif args[0] in ['plain', 'lines']:
        with open(args[2], 'w') as script_file:
            for speech in dom.getElementsByTagName('SPEECH'):
                text = get_text(speech.getElementsByTagName('LINE'))
                script_file.write(text + (' ' if args[0] == 'plain' else '\n'))
    else:
        print('Unknown output specifier')
        fail()


if __name__ == '__main__':
    main(sys.argv[1:])
