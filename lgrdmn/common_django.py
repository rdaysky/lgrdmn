# File encoding: UTF-8

from django.conf import settings

from django.contrib                import auth
from django.core.exceptions        import *
from django.core.serializers.json  import DjangoJSONEncoder
from django.core.urlresolvers      import reverse as django_reverse
from django.db                     import transaction
from django.dispatch               import receiver
from django.http                   import HttpResponse, HttpResponsePermanentRedirect, HttpResponseRedirect, HttpResponseForbidden, HttpResponseNotFound, HttpResponseServerError
from django.http                   import Http404
from django.shortcuts              import get_object_or_404, render_to_response
from django.template               import RequestContext
from django.template.loader        import render_to_string
from django.utils.encoding         import force_text
from django.views.decorators.csrf  import csrf_exempt

import django.conf.urls
import django.contrib.auth.models
import django.contrib.staticfiles.finders
import django.db
import django.db.models
import django.db.models.manager
import django.db.models.query
import django.forms
import django.utils._os
import django.utils.module_loading

from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage
import email.utils

import importlib
import inspect
import itertools
import json
import logging
import os
import sys
import uuid

from .common_all import *

try:
    conf = importlib.import_module(settings.LGRDMN_CONF)
except Exception:
    try:
        conf = importlib.import_module(settings.INSTALLED_APPS[-1] + ".conf")
    except Exception:
        conf = None

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

class CharFieldIgnoringChoices(django.forms.CharField):
    def __init__(self, coerce=None, choices=None, *a, **k):
        super(CharFieldIgnoringChoices, self).__init__(*a, **k)

class EnumField(django.db.models.Field):
    __metaclass__ = django.db.models.SubfieldBase

    empty_strings_allowed = False
    description = "Enum value"

    def __init__(self, enum_class, display_attribute=None, *a, **k):
        self.enum_class = enum_class
        self.display_attribute = display_attribute

        self.explicit_choices = bool(k.get("choices", None))
        if not self.explicit_choices:
            k["choices"] = [(x, getattr(x, display_attribute or "_name_")) for x in enum_class]

        return super(EnumField, self).__init__(*a, **k)

    def deconstruct(self):
        name, path, args, kwargs = super(EnumField, self).deconstruct()
        kwargs["enum_class"] = self.enum_class
        if self.display_attribute:
            kwargs["display_attribute"] = self.display_attribute
        if not self.explicit_choices:
            kwargs.pop("choices", None)
        return name, path, args, kwargs

    def get_internal_type(self):
        return "IntegerField"

    def to_python(self, value):
        if value is None:
            return None

        if isinstance(value, basestring):
            if value.isdigit():
                return self.enum_class(int(value))

            prefix, dot, name = value.rpartition(".")
            if dot == ".":
                assert prefix == self.enum_class.__name__
            return self.enum_class[name]
        return self.enum_class(value)

    def get_prep_value(self, value):
        value = super(EnumField, self).get_prep_value(value)
        if value is None:
            return None
        if isinstance(value, int):
            return value
        return value._value_

    def formfield(self, **kwargs):
        return super(EnumField, self).formfield(**combine_dicts(
            kwargs,
            form_class=django.forms.CharField,
            choices_form_class=CharFieldIgnoringChoices,
        ))

class ExistingStaticFilesFinder(django.contrib.staticfiles.finders.BaseFinder):
    """ A static files finder for files already existing under STATIC_ROOT.  """
    def find(self, path, all=False):
        fs_path = django.utils._os.safe_join(settings.STATIC_ROOT, path)
        if os.path.exists(fs_path):
            return [fs_path] if all else fs_path
        return [] # sic (even if all=False)

    def list(self, ignore_patterns):
        return []

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

def _limit_offset_int(items, limit=None, offset=0):
    if limit is None:
        return items
    assert limit >= 0 and offset >= 0
    return items[offset : limit + offset]

def get_limit_offset(request=None, extra=0):
    if request is None or "limit" not in request.REQUEST:
        return None, 0

    limit  = int(request.REQUEST["limit"]) + extra
    offset = int(request.REQUEST.get("offset", 0))

    return limit, offset

def _limit_offset_request(items, request, extra=0):
    return _limit_offset_int(items, *get_limit_offset(request, extra))

def limit_offset(items, *a, **k):
    if a:
        assert not k
        return _limit_offset_int(items, *a)

    if k:
        assert not a
        if "limit" in k:
            return _limit_offset_int(items, **k)
        return _limit_offset_request(items, **k)

    return items


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

_uidbwhatever = None
def uidb36or64():
    global _uidbwhatever
    if _uidbwhatever is not None:
        return _uidbwhatever

    import django.http
    import django.contrib.auth.views

    def is_bad_argument(name):
        try:
            django.contrib.auth.views.password_reset_confirm(django.http.HttpRequest(), **{name: None})
        except TypeError, e:
            if "got an unexpected keyword argument" in e.message:
                return True
            return False
        except:
            return False
        else:
            return False

    is_uidb36_bad = is_bad_argument("uidb36")
    is_uidb64_bad = is_bad_argument("uidb64")

    assert is_uidb36_bad != is_uidb64_bad

    _uidbwhatever = "uidb36" if is_uidb64_bad else "uidb64"
    return _uidbwhatever

class Deconstructor(object):
    def deconstruct(self):
        return "{}.{}".format(self.__class__.__module__, self.__class__.__name__), (), {}

    def __eq__(self, anything):
        return True

class DeconstructibleFunction(Deconstructor):
    def __init__(self, function):
        self.function = function

    def __call__(self, *a, **k):
        return self.function(*a, **k)

def deconstructible_function(function):
    return DeconstructibleFunction(function)

def get_qs(some_db_set):
    if isinstance(some_db_set, django.db.models.query.QuerySet):  return some_db_set
    if isinstance(some_db_set, django.db.models.manager.Manager): return some_db_set.all()
    if issubclass(some_db_set, django.db.models.Model):           return some_db_set._default_manager.all()

    assert False, "get_qs: unexpected {}".format(some_db_set if inspect.isclass(some_db_set) else type(some_db_set))

def get_manager(some_db_set):
    if isinstance(some_db_set, django.db.models.query.QuerySet):  return some_db_set.model._default_manager
    if isinstance(some_db_set, django.db.models.manager.Manager): return some_db_set
    if issubclass(some_db_set, django.db.models.Model):           return some_db_set._default_manager

    assert False, "get_manager: unexpected {}".format(some_db_set if inspect.isclass(some_db_set) else type(some_db_set))

def get_object_or_none(__objects, *args, **kwargs):
    qs = get_qs(__objects)

    try:
        return qs.get(*args, **kwargs)
    except qs.model.DoesNotExist:
        return None

def create_or_ignore(__objects, **kwargs):
    manager = get_manager(__objects)
    try:
        with transaction.atomic():
            return manager.create(**kwargs)
    except django.db.IntegrityError:
        return None

def lookup_from_key(key, model):
    res = key.copy()
    for f in model._meta.fields:
        if f.attname in res:
            res[f.name] = res.pop(f.attname)
    return res

def create_from_key(key):
    return { k: v for k, v in key.items() if "__" not in k }

def create_or_delete(objects, create, key, data={}, ignore=False, ignore_create=None, ignore_delete=None):
    if create:
        create_params = combine_dicts(
            create_from_key(key),
            data,
        )

        if coalesce(ignore_create, ignore):
            return create_or_ignore(objects, **create_params) is not None
        else:
            get_manager(objects).create(**create_params)
            return True
    else:
        qs = get_qs(objects)
        c = qs.filter(**key).delete()
        if c > 0:
            return True
        elif coalesce(ignore_delete, ignore):
            return False
        else:
            raise qs.model.DoesNotExist()

def create_or_replace(objects, key, data={}, condition=None):
    # https://code.djangoproject.com/attachment/ticket/3182/3182.update_or_create-only.3.diff

    assert len(key) > 0
    qs = get_qs(objects)

    qs._for_write = True

    with transaction.atomic():
        try:
            instance = qs.get(**lookup_from_key(key, model=qs.model))

            if condition:
                if not condition(instance):
                    return instance, False

            created = False
        except qs.model.DoesNotExist:
            instance = qs.model(**create_from_key(key))
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

def xar_response(url):
    response = HttpResponse()
    if "Content-Type" in response:
        del response["Content-Type"]
    response["X-Accel-Redirect"] = url
    return response

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
        return None

    fast_login(request, user, remember=remember)

    return user

def set_field_widget_attribute(form, field, name, value):
    widget = form.base_fields[field].widget
    if widget.attrs is None:
        widget.attrs = {}
    widget.attrs[name] = value

def json_requested(request):
    http_accept = request.META.get("HTTP_ACCEPT", "")
    json_accepted, html_accepted = [(t in http_accept) for t in
        ("application/json", "text/html")] # TODO: perhaps parse the field better

    if conf.json.re_api_json_url.match(request.path):
        return True

    if json_accepted:
        return True

    if "json_output" in request.REQUEST:
        return True

    if html_accepted:
        return False

    #if request.is_ajax():
    #    return True

    return False

def item_or_404(hash, subscript, message=None):
    """ Returns hash[subscript], converting KeyError to Http404. """
    try:
        return hash[subscript]
    except KeyError:
        raise Http404(coalesce(message, "No item corresponds to given subscript."))

_GAI_NO_DEFAULT = object()
def get_any_item(items, *names, **kwargs):
    default = kwargs.pop("default", _GAI_NO_DEFAULT)
    assert not kwargs

    for name in names:
        if name in items:
            return items[name]

    if default is not _GAI_NO_DEFAULT:
        return default

    raise KeyError()

class AuthenticationRequired(Exception):
    """ Stores the reason of the error and what to show on the page. """
    def __init__(self, error=None, username_hint=None):
        self.error = error
        self.username_hint = username_hint

    def __str__(self):
        return self.error

class InvalidForm(Exception):
    """ Stores information about failed form validation. """
    def __init__(self, form):
        self.form = form

    def __str__(self):
        try:
            return u"; ".join([u"* %s: %s" % (k, u"; ".join([force_unicode(i) for i in v])) for k, v in self.form.errors.items()])
        except:
            return "[Some errors]"

class InvalidSubmission(Exception):
    """ Stores information about a submission failure that’s not related to a field. """
    def __init__(self, error_key):
        self.error_key = error_key

class ApiError(Exception):
    def __init__(self, reason, data={}):
        self.reason = reason
        self.data = data

def enforce_valid(form):
    """ Requres is_valid(). Can raise InvalidForm. """
    if not form.is_valid():
        raise InvalidForm(form)
    return form

def enforce_update(values, instance, fields=None, exclude=None, files={}):
    from django.db.models.fields.files import FileField
    unaffected_field_names = { f.name for f in instance._meta.fields }
    for f in instance._meta.fields:
        if fields is not None and f.name not in fields:
            continue
        if exclude is not None and f.name in exclude:
            continue

        data = files if isinstance(f, FileField) else values

        if f.name not in data:
            continue

        setattr(instance, f.attname, data[f.name])
        unaffected_field_names.remove(f.name)

    try:
        instance.full_clean(exclude=unaffected_field_names, validate_unique=False)
    except ValidationError as e:
        class FakeForm:
            errors = e.message_dict
        raise InvalidForm(FakeForm())
    return instance

def attempted_post(request):
    return request.POST if hasattr(request, "_failed_form") else None

def form_errors(request):
    """ Returns request._failed_form.errors | request._failed_form | None. """
    if not hasattr(request, "_failed_form"):
        return {}

    ff = request._failed_form

    return getattr(ff, "errors", ff)

post_by_action = {}
def post_handler(action, next=None, stay=False, atomic=True, validate_session=False): # Decorator
    """ Registers a handler for a given POST action, validates session, issues a redirect. """
    def ph_outer(view):
        def ph_inner(request):
            if request.method != "POST":
                raise Http404("Only POST can be used for this kind of query") # TODO: HTTP 405?

            if validate_session:
                if request.POST.get(conf.session_validation_token_post_name, None) != hash_secret(request.COOKIES[settings.SESSION_COOKIE_NAME]):
                    raise Http404("CSRF") # TODO: hax0rz

            wrapped_view = transaction.atomic(view) if atomic else view

            result = wrapped_view(request)
            if isinstance(result, HttpResponse):
                return result

            if stay or conf.submit_stay in request.POST:
                result = None
            elif next:
                result = next(request) or result
            return result or request.path # TODO: HttpResponseRedirect here and not in post_middleware?

        assert action not in post_by_action
        post_by_action[action] = ph_inner
        return ph_inner
    return ph_outer

def access_restricted(_view=None, require_active=True, require_staff=False): # Decorator
    """ Requires that user be authenticated. """
    def ar_outer(_view):
        def ar_inner(request, *args, **kwargs):
            if request.user is None or not request.user.is_authenticated():
                raise AuthenticationRequired()

            if require_active and not request.user.is_active:
                raise AuthenticationRequired("user_inactive")

            if require_staff and not request.user.is_staff:
                raise AuthenticationRequired("lacks_permission")

            return _view(request, *args, **kwargs)
        return ar_inner

    if _view:
        return ar_outer(_view)
    return ar_outer

def admin_autoregister_module(module_name):
    import inspect
    import django.db.models
    import django.contrib.admin.sites

    try:
        ModelAdmin = sys.modules["django.contrib.gis.admin"].OSMGeoAdmin
    except Exception:
        ModelAdmin = None

    module_dict = sys.modules[module_name].__dict__
    for name, model in module_dict.items():
        if inspect.isclass(model) and model.__module__ == module_name and issubclass(model, django.db.models.Model) and not model._meta.abstract:
            try:
                django.contrib.admin.site.unregister(model)
            except django.contrib.admin.sites.NotRegistered:
                pass

            try:
                django.contrib.admin.site.register(model, module_dict.get("%sAdmin" % name, ModelAdmin))
            except django.contrib.admin.sites.AlreadyRegistered:
                pass
