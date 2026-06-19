
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path

from detection import views
from detection.forms import EmailAuthenticationForm


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.home, name="home"),
    path("accounts/register/", views.register, name="register"),
    path(
        "accounts/login/",
        LoginView.as_view(
            template_name="accounts/login.html",
            authentication_form=EmailAuthenticationForm,
        ),
        name="login",
    ),
    path("accounts/logout/", LogoutView.as_view(), name="logout"),
    path("profile/", views.profile, name="profile"),
    path("detect/", views.detect, name="detect"),
    path("runs/<int:run_id>/", views.run_detail, name="run_detail"),
    path("history/", views.history, name="history"),
    path("stats/", views.stats, name="stats"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
