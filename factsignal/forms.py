from django import forms


class CoverForm(forms.Form):
    input_text = forms.CharField()
