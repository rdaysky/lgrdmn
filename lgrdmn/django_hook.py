# File encoding: UTF-8

"""
This file is for objects that are not meant to be called directly,
but rather added to processing chains via settings.py.
For example, custom middleware and context processors go here.
"""

from lgrdmn.common import *

import django.core.urlresolvers
django.core.urlresolvers.get_resolver(None).urlconf_module # Preload modules to populate post_by_action.

import django.views.debug
import django.utils.cache
from django.core.mail.backends.base import BaseEmailBackend

import django.contrib.sessions.middleware
import django.contrib.auth.middleware
import django.contrib.auth.models

import base64
import datetime
import subprocess
import traceback

logger = logging.getLogger(__name__)

def do_submission(request, handler=None):
    handler = handler or get_submission_handler(request)
    try:
        function = post_by_action[handler]
    except KeyError: # TODO: proper internal errors.
        logger.debug("bad handler \"{}\"".format(handler))
        return HttpResponseNotFound()

    return function(request)

def get_submission_handler(request):
    if conf.submission_action_post_name in request.POST:
        return request.POST[conf.submission_action_post_name]

    m = conf.submission_action_url_template.match(request.path)
    if m:
        return m.group(1)

    return None

def response_authentication_required(request, exception):
    if request.META.get("HTTP_X_PREFERRED_AUTHENTICATION", "") == "basic":
        response = HttpResponse("Authorization required", content_type="text/plain")
        response["WWW-Authenticate"] = "Basic realm=\"%s\"" % conf.basic_auth_realm
        response.status_code = 401
        return response
    if json_requested(request):
        return json_response({
            "success": False,
            "reason": "authentication_required",
            "error": exception.error,
        }, status=401, request=request)
    return HttpResponseForbidden()
    #return generic.login(request, exception.error, exception.username_hint) # TODO: put back if we ever have actual auth

def try_process_exception(request):
    json_output = json_requested(request)

    logger.exception("Error in {} {}".format(request.method, request.POST.get(conf.submission_action_post_name) or request.path))

    try:
        try:
            raise

        except InvalidForm as e:
            if json_output:
                return json_response({
                    "success": False,
                    "reason": "invalid_form",
                    "errors": e.form.errors,
                }, status=403, request=request)

            request._failed_form = e.form
            return None

        except InvalidSubmission as e:
            if json_output:
                return json_response({
                    "success": False,
                    "reason": "invalid_submission",
                    "error": e.error_key,
                }, status=403, request=request)

            request._submission_error_key = e.error_key
            return None

        except AuthenticationRequired as e:
            return response_authentication_required(request, e)

        except ApiError as e:
            if json_output:
                return json_response(combine_dicts({
                    "success": False,
                    "reason": e.reason,
                }, e.data), status=403, request=request)
            else:
                raise

        # Working around Django bug #6094. TODO: FIXME all the rest of the function is a dirty hack.
        except Http404 as e:
            if settings.DEBUG:
                from django.views import debug
                return debug.technical_404_response(request, e)
            return HttpResponseNotFound()

    except Exception as e:
        if json_output:
            response = {
                "success": False,
                "reason": "exception",
            }
            if settings.DEBUG:
                response.update({
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                })
            return json_response(response, status=500, request=request)

        if not settings.DEBUG or settings.DEBUG_PROPAGATE_EXCEPTIONS:
            raise

        return django.views.debug.technical_500_response(request, *sys.exc_info())

@json_encoder_class(DjangoJSONEncoder)
def JsonEncoder(o, super):
    if hasattr(o, "to_struct"):
        return o.to_struct()

    if isinstance(o, (datetime.datetime, datetime.time)):
        o = o.replace(microsecond=0)

    return super(o)

class HostUrlMiddleware(object):
    def process_request(self, request):
        host = request.META.get("HTTP_HOST")
        if not host:
            return

        if host.endswith(":80"):
            host = host[:-3]

        if host in settings.HOST_URLCONF:
            request.urlconf = settings.HOST_URLCONF[host]

    def process_response(self, request, response):
        if getattr(request, "urlconf", None):
            django.utils.cache.patch_vary_headers(response, ("Host",))
        return response


class FastSessionMiddleware(django.contrib.sessions.middleware.SessionMiddleware):
    def process_request(self, request):
        m = conf.re_api_version.match(request.path)
        request._api_version = int(m.group(1)) if m else None

        if request._api_version is not None and settings.SESSION_COOKIE_NAME not in request.COOKIES:
            return

        super(FastSessionMiddleware, self).process_request(request)

    def process_response(self, request, response):
        if getattr(request, "_api_version", None) is not None:
            return response

        return super(FastSessionMiddleware, self).process_response(request, response)

class FastAuthenticationMiddleware(auth.middleware.AuthenticationMiddleware):
    def process_request(self, request):
        if hasattr(request, "session"):
            return super(FastAuthenticationMiddleware, self).process_request(request)

        request.user = auth.models.AnonymousUser()

    assert not hasattr(auth.middleware.AuthenticationMiddleware, "process_response")

class PostMiddleware(object):
    """ Attempts to dispatch the request based on POST[submission_action_post_name]. On invalid_form, proceeds with GET to same URL. """
    def process_request(self, request):
        if request.method != "POST":
            return None

        submission_handler = get_submission_handler(request)
        if not submission_handler:
            return None

        try:
            try:
                response_or_target = do_submission(request, submission_handler)

                if isinstance(response_or_target, HttpResponse):
                    return response_or_target
                elif json_requested(request) or isinstance(response_or_target, JsonResponse):
                    return json_response(combine_dicts({
                        "success": True,
                        "target": full_url(response_or_target) if isinstance(response_or_target, basestring) else None,
                    }, response_or_target.struct if isinstance(response_or_target, JsonResponse) else {}), request=request)
                elif isinstance(response_or_target, basestring):
                    return HttpResponseRedirect(response_or_target)
                else:
                    raise Exception("POST view functions must return HttpResponse/JsonResponse instances or URL strings; got %s instead" % type(response_or_target).__name__)
            except:
                if settings.DEBUG:
                    logger.debug("POST failure: %s" % request.POST, exc_info=True)

                # Working around Django bug #6094.
                if django.db.transaction.is_dirty():
                    django.db.transaction.rollback()
                    django.db.transaction.leave_transaction_management()
                raise
        except Exception:
            return try_process_exception(request)

class MiscMiddleware(object):
    def process_exception(self, request, exception):
        try:
            return try_process_exception(request)
        except:
            return

class AuthMiddleware(object):
    def process_request(self, request):
        if not hasattr(request, "user"):
            raise ImproperlyConfigured("Need AuthenticationMiddleware before this middleware")

        if settings.DEBUG and request.user.is_superuser and "impersonate" in request.GET:
            try:
                user = auth.models.User.objects.get(username=request.GET["impersonate"])
            except auth.models.User.DoesNotExist:
                pass
            else:
                set_auth_backend(user)
                request.user = user # But keep request.session. Beware of that.
                return

        if "HTTP_AUTHORIZATION" not in request.META:
            return

        auth_data = request.META["HTTP_AUTHORIZATION"].split()
        if len(auth_data) < 2:
            return

        if auth_data[0].lower() == "basic":
            username, password = base64.b64decode(auth_data[1]).split(":")
            try:
                u = authenticate_and_login(request, username, password)
                if u is None:
                    raise AuthenticationRequired(error="wrong_credentials", username_hint=username)
            except AuthenticationRequired, e:
                return response_authentication_required(request, e)
            else:
                request.csrf_processing_done = True

    def process_exception(self, request, exception):
        if isinstance(exception, AuthenticationRequired):
            return response_authentication_required(request, exception)

class CookieRemovalMiddleware(object):
    def process_response(self, request, response):
        if hasattr(request, "session") and "HTTP_AUTHORIZATION" in request.META:
            response.delete_cookie(settings.SESSION_COOKIE_NAME)
        return response

class StandaloneServerMiddleware(object):
    def __init__(self):
        if not is_server_standalone():
            raise MiddlewareNotUsed()

    def process_response(self, request, response):
        if "MSIE" in request.META.get("HTTP_USER_AGENT", ""):
            response["X-UA-Compatible"] = "IE=Edge,chrome=1"

        return response

class DisableCsrfMiddleware(object):
    def process_request(self, request):
        setattr(request, "_dont_enforce_csrf_checks", True)

class UsernameOrEmailAuthenticationBackend(auth.backends.ModelBackend):
    def authenticate(self, username=None, password=None):
        if not username:
            return None

        lookup = {("email" if "@" in username else "username"): username.lower()}
        try:
            user = auth.models.User.objects.get(**lookup)
            if user.check_password(password):
                return user
        except auth.models.User.DoesNotExist:
            return None

class SendmailBackend(BaseEmailBackend):
    def __init__(self, fail_silently=False, **kwargs):
        super(SendmailBackend, self).__init__(fail_silently=fail_silently)

    def open(self):
        return True

    def close(self):
        pass

    def send_messages(self, email_messages):
        c_sent = 0
        for message in email_messages:
            if self._send_one(message):
                c_sent += 1
        return c_sent

    def _send_one(self, email_message):
        try:
            p = subprocess.Popen(["/usr/sbin/sendmail", "-f", email_message.from_email, "-t"], stdin=subprocess.PIPE)
            p.communicate(email_message.message().as_string())
            return p.returncode == 0
        except:
            if self.fail_silently:
                return False
            raise

def template_attributes(request):
    host = request.META["HTTP_HOST"].split(":")[0]

    return dict(
        api_keys = combine_dicts(
            dict((name, key_for_host(keys, host)) for name, keys in conf.host_api_keys.__dict__.items() if not name.startswith("__")),
            conf.api_keys.__dict__,
        ),
        conf = conf.web,
        FULL_MEDIA_URL = full_url(settings.MEDIA_URL, host=settings.HOST_MEDIA),
    )

def _mp():
    import django

    @monkey_patch_replace_method(django.forms.widgets.CheckboxInput)
    def value_from_datadict(orig, self, data, files, name):
        if name not in data:
            # A missing value means False because HTML form submission does not
            # send results for unselected checkboxes.
            return False
        value = data.get(name)
        # Translate true and false strings to boolean values.
        values = {
            "1": True,
            "true": True,
            "0": False,
            "false": False,
        }
        if isinstance(value, six.string_types):
            value = values.get(value.lower(), value)
        return bool(value)

_mp()

