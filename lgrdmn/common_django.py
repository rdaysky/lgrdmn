# File encoding: UTF-8

from django.conf import settings

from django.contrib                import auth
from django.core.exceptions        import ObjectDoesNotExist, ImproperlyConfigured, MiddlewareNotUsed
from django.core.serializers.json  import DjangoJSONEncoder
from django.core.urlresolvers      import reverse as django_reverse
from django.db                     import transaction
from django.dispatch               import receiver
from django.http                   import HttpResponse, HttpResponseRedirect, HttpResponsePermanentRedirect, HttpResponseServerError, HttpResponseForbidden
from django.http                   import Http404
from django.shortcuts              import get_object_or_404, render_to_response
from django.template               import RequestContext
from django.template.loader        import render_to_string
from django.utils.encoding         import force_text
from django.views.decorators.csrf  import csrf_exempt

import django.conf.urls
import django.contrib.auth.models
import django.db
import django.db.models.manager
import django.db.models.query
import django.forms
import django.utils.module_loading

from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage
import email.utils

import sys
import os
import itertools
import uuid

import json

from .common_all import *

try:
    import django.contrib.auth.hashers
except ImportError:
    def django_unusable_password():
        return auth.models.UNUSABLE_PASSWORD
else:
    def django_unusable_password():
        return auth.hashers.make_password(None)

def str_uuid():
    return str(uuid.uuid4())

def lex_str(x):
    return "%08d" % x if isinstance(x, (int, long)) else str(x)

class StrippingCharField(django.forms.CharField):
    def to_python(self, value):
        res = super(StrippingCharField, self).to_python(value)
        return res.strip() if isinstance(res, basestring) else res

def objects_extra(Manager=django.db.models.Manager, QuerySet=django.db.models.query.QuerySet):
    def oe_inner(Mixin, Manager=django.db.models.Manager, QuerySet=django.db.models.query.QuerySet):
        class MixinManager(Manager, Mixin):
            class MixinQuerySet(QuerySet, Mixin):
                pass

            def get_query_set(self):
                return self.MixinQuerySet(self.model, using=self._db)

        return MixinManager()

    if issubclass(Manager, django.db.models.Manager):
        return lambda Mixin: oe_inner(Mixin, Manager, QuerySet)
    else:
        return oe_inner(Mixin=Manager)

@overload
def limit_offset(items, limit, offset=0):
    assert limit >= 0 and offset >= 0
    return items[offset : limit + offset]

@limit_offset.overload
def limit_offset_request(items, request):
    if request is None or "limit" not in request.REQUEST:
        return items

    limit  = int(request.REQUEST["limit"])
    offset = int(request.REQUEST.get("offset", 0))

    return limit_offset(items, limit=limit, offset=offset)

def request_bool(REQUEST, name, missing=False, blank=True):
    if name not in REQUEST:
        return missing
    return {
        "": blank,
        "1": True,
        "0": False,
    }[REQUEST[name]]

def flatten_choices(choices):
    """ Returns choices without groups (if any). """
    return itertools.chain(*(v if isinstance(v, (tuple, list)) else [(k, v)] for k, v in choices))

def url_from_view(view, _urlconf=None, **kwargs):
    return django_reverse(view, urlconf=_urlconf, kwargs=dict([(k, v) for (k, v) in kwargs.items() if v is not None]))

def full_url(url, host=settings.HOST_DJANGO, scheme="http"):
    return "%s://%s%s" % (scheme, host, url) if url.startswith("/") and host else url

def url_patterns_without_spaces(prefix, *args): # would have used (?x) if that had allowed reversing
    """ Calls django.conf.urls.patterns after removing spaces from regexps. """
    def without_spaces(pattern):
        if isinstance(pattern, (list, tuple)):
            return [pattern[0].replace(" ", "")] + list(pattern[1:])
        else:
            return pattern
    return django.conf.urls.patterns(prefix, *[without_spaces(i) for i in args])

def void():
    return None

class deconstructible_function:
    def __init__(self, function):
        self.function = function

    def deconstruct(self):
        return (void.__module__ + ".void", (), {})

    def __call__(self, *a, **k):
        return self.function(*a, **k)


def get_qs(some_db_set):
    if isinstance(some_db_set, django.db.models.query.QuerySet):
        return some_db_set
    elif isinstance(some_db_set, django.db.models.manager.Manager):
        return some_db_set.all()
    else:
        return some_db_set._default_manager.all()

def get_object_or_none(__objects, *args, **kwargs):
    qs = get_qs(__objects)

    try:
        return qs.get(*args, **kwargs)
    except qs.model.DoesNotExist:
        return None

def create_or_replace(objects, key, data, condition=None):
    # https://code.djangoproject.com/attachment/ticket/3182/3182.update_or_create-only.3.diff

    qs = get_qs(objects)
    assert key and data

    lookup = key.copy()
    for f in qs.model._meta.fields:
        if f.attname in lookup:
            lookup[f.name] = lookup.pop(f.attname)

    qs._for_write = True

    with xact():
        try:
            instance = qs.get(**lookup)

            if condition:
                if not condition(instance):
                    return instance, False

            created = False
        except qs.model.DoesNotExist:
            params = dict([(k, v) for k, v in key.items() if "__" not in k])
            instance = qs.model(**params)
            created = True

        for attname, value in data.items():
            setattr(instance, attname, value)

        instance.save(force_insert=created, using=qs.db)
        return instance, created

def values_list_flat(object, field):
    if not hasattr(object, "values_list"):
        raise Exception("django.db.models.fields.related... required")
    return [k for k, in object.values_list(field)]

def url_thumbnail(url_original, dimensions):
    return url_original.replace("/original.", "/thumbnail_%sx%s." % dimensions, 1)

def make_message(From, To, Subject, Text, headers={}, attach=[]):
    """ Returns an email.message.Message object populated with headers and data. """
    m = email.mime.multipart.MIMEMultipart()
    m["To"] = To
    m["From"] = From
    m["Subject"] = Subject
    for k, v in headers.items():
        m[k] = v
    m.attach(email.mime.text.MIMEText(Text))
    for name, (type1, type2), bytes in attach:
        part = email.mime.base.MIMEBase(type1, type2)
        part.set_payload(bytes)
        email.encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=\"%s\"" % name)
        m.attach(part)
    return m

def save_uploaded_file(django_file, path):
    try:
        target = open(path, "wb")
        for chunk in django_file.chunks():
            target.write(chunk)
        target.close()
    except:
        try:
            os.unlink(path)
        except:
            pass
        raise

def is_server_standalone():
    return len(sys.argv) >= 2 and sys.argv[1] == "runserver"

def struct_instance(instance, fields=None, exclude=[], display=True, rel=True):
    def field(instance, name):
        return dict(value=getattr(instance, name), display=getattr(instance, "get_%s_display" % name)()) \
            if display and hasattr(instance, "get_%s_display" % name) \
            else getattr(instance, name)

    return dict([
        (f.name, field(instance, f.attname))
        for f in instance._meta.fields
        if (fields is None or f.name in fields)
            and f.name not in exclude
            and (rel or not f.rel)
    ])

class LazySave(object):
    def __init__(self, function):
        self.function = function
        self.value = None

    def get(self):
        if self.value is None:
            self.value = self.function()
            assert self.value is not None
        return self.value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.value is not None and exc_type is None:
            self.value.save()

try:
    django_import = django.utils.module_loading.import_string
except:
    django_import = django.utils.module_loading.import_by_path

_JsonEncoder = None
def get_json_encoder_class():
    global _JsonEncoder
    if _JsonEncoder is None:
        _JsonEncoder = django_import(settings.JSON_ENCODER) if hasattr(settings, "JSON_ENCODER") else DjangoJSONEncoder
    return _JsonEncoder

def json_params(request):
    """ Gets and validates parameters for json_response by inspecting request. """
    http_accept = request.META.get("HTTP_ACCEPT", "")
    json_accepted, plaintext_accepted = [(t in http_accept) for t in
        ("application/json", "text/plain")] # TODO: perhaps parse the field better

    params = {}
    if "jsonp" in request.GET:
        params["jsonp"] = request.GET["jsonp"]
    if request.path.endswith(".txt") or "json_plaintext" in request.GET or (json_accepted and plaintext_accepted):
        params["plaintext"] = True
    if request.GET.get("json_content_type") in ("application/json", "text/plain", "text/html"):
        params["content_type"] = request.GET["json_content_type"]
    return params

def json_response(data=None, raw=None, jsonp="", plaintext=False, content_type=None, request=None, status=200):
    """ Returns HttpResponse("jsonp(json)"), sets Content-Type to application/json or text/plain depending on plaintext. """
    if request:
        return json_response(data, raw, status=status, **json_params(request))
    json = raw if raw is not None else write_json(data, pretty=plaintext, encoder_class=get_json_encoder_class())
    return HttpResponse("%s(%s)" % (jsonp, json) if jsonp and conf.json.regex_jsonp.match(jsonp) else json,
        content_type=(content_type or ("text/plain" if plaintext else "application/json")),
        status=status,
    )

class JsonResponse(object):
    def __init__(self, struct):
        self.struct = struct

def fast_login(request, user, remember=False):
    if not hasattr(request, "session"):
        assert not remember
        request.user = user
        return

    auth.login(request, user) # Flushes session if user changes

    # set_expiry: 0 = session, None = settings.SESSION_COOKIE_AGE
    request.session.set_expiry(None if remember else 0)

def set_auth_backend(user, backend=None):
    if not backend:
        backend = auth.get_backends()[0]
    if not isinstance(backend, basestring):
        backend = "%s.%s" % (backend.__module__, backend.__class__.__name__)
    user.backend = backend
    return user

def authenticate_and_login(request, username, password, remember=False):
    user = auth.authenticate(username=username, password=password)
    if user is None:
        raise AuthenticationRequired(error="wrong_credentials", username_hint=username)

    fast_login(request, user, remember=remember)

    return user

def set_field_widget_attribute(form, field, name, value):
    widget = form.base_fields[field].widget
    if widget.attrs is None:
        widget.attrs = {}
    widget.attrs[name] = value
