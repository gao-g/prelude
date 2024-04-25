from ipywidgets import HTML, VBox, widgets, Layout, HBox, interactive
import diff_match_patch as dmp_module
import json
import pandas as pd

class Diff:
    def __init__(self, old, new, title_old, title_new):
        self.dmp = dmp_module.diff_match_patch()
        self.diff = self.dmp.diff_main(old, new)
        self.dmp.diff_cleanupSemantic(self.diff)
        self.title_old = title_old
        self.title_new = title_new

    @property
    def inplace(self):
        return HTML(f'<header><h3>{self.title_old} vs {self.title_new}</h3></header>{self.dmp.diff_prettyHtml(self.diff)}')
    
    @property
    def side_by_side(self):
        def _translate(flag, data):
            text = data.translate(str.maketrans({"&": "&amp;","<": "&lt;",">": "&gt;","\n": "<br>"}))
            if flag == dmp_module.diff_match_patch.DIFF_DELETE:
                return flag, f'<del style="background:#ffe6e6;">{text}</del>'
            if flag == dmp_module.diff_match_patch.DIFF_INSERT:
                return flag, f'<ins style="background:#e6ffe6;">{text}</ins>'
            elif flag == dmp_module.diff_match_patch.DIFF_EQUAL:
                return flag, f'<span>{text}</span>'
        translated = [_translate(flag, data) for (flag, data) in self.diff]
        old = "".join([t for (f, t) in translated if f != dmp_module.diff_match_patch.DIFF_INSERT])
        new = "".join([t for (f, t) in translated if f != dmp_module.diff_match_patch.DIFF_DELETE])
        return HBox([
            HTML(f'<header><h3>{self.title_old}</h3></header>{old}'),
            HTML(f'<header><h3>{self.title_new}</h3></header>{new}')], layout = Layout(justify_content = 'space-around'))

        
def dict2html(d):
    def _with_br(text):
        return str(text).replace('\n', '<br/>')
    table = "".join([f"<tr><td>{k}</td><td>{_with_br(v)}</td></tr>" for k,v in d.items()])
    return HTML(value=f'<table><body><tbody>{table}</tbody></tbody></table>')

class Row(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return f'Cost: {self["cost"]}--------{self["message"][:128]}...'

    def view(self, mode = 'side-by-side'):
        def _html(section):
            return HTML(value=f"<header><h3>{section}</h3></header>{self[section]}")
    
        def _view(mode):
            if mode == 'raw':
                return dict2html(self)
            diff = Diff(self["completion"], self["edited"], 'completion', 'edited')
            view = {
                "comment": self['comment'],
                "log p(completion|message)": sum(self['completion_logprobs']),
                "tokens(completion)": self['completion_token_count'],                
                "log p(edited|message)": sum(self['edited_logprobs']),
                "tokens(edited)": self['edited_token_count'],
                "preference.inference": self['preference_inference'],
                "preference.groundtruth": self['preference_groundtruth'],
            }
            conversation = self.get('conversation', [])
            for c in conversation:
                c['q'] = c['q'] if isinstance(c['q'], str) else c['q'][1]['content']
            return VBox(
                children = [
                    _html("message"),
                    diff.inplace if mode == 'inplace' else diff.side_by_side,
                    dict2html(view),
                    widgets.Accordion(
                        children = [dict2html(i) for i in conversation],
                        titles=[i['q'][:128] if isinstance(i['q'], str) else i['q'][1]['content'][:128] for i in conversation])
                ])
        return _view(mode)


class Logs(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            return Logs(map(lambda line: Row(json.loads(line)), f))

    def __getitem__(self, i):
        result = super().__getitem__(i)
        return Logs(result) if isinstance(i, pd.core.series.Series) else result
    
    def sort_values(self, *args, **kwargs):
        return Logs(super().sort_values(*args, **kwargs))
    
    def view(self, mode = 'side-by-side'):
        i_row = [(i, Row(row)) for i, row in self.iterrows()]
        return widgets.Accordion(
            children=[row.view(mode) for i, row in i_row],
            titles = [f'{i}. {str(row)}' for i, row in i_row])
