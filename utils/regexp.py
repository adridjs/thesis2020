class RegExp(object):
    def __init__(self):
        self.doc_properties = r'<doc docid="((?:\w*\s*)*)" wpid="(\w*)" language="(\w*)" gender="(\w*)">'
        self.segment = r'<seg id="\w+">(\[.+)<\\seg>'
        self.title = r'(?:<title>((?:\w*\s*)*)<\/title>)?'

        self.doc_wise = f'{self.title}\\n {self.segment}'
