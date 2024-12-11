from django.urls import path
from .views import (
    AdminDashboardView, AdminUserDataView, AdminAddPlanView, 
    AddPaymentPlanView, ViewPaymentPlansView, EditPaymentPlanView,EditPaymentPlanPageView
)

urlpatterns = [
    path('admin/admin_page/', AdminDashboardView.as_view(), name='admin_dashboard'),
    path('admin/user_data/', AdminUserDataView.as_view(), name='admin_user_data'),
    path('admin/add_plan/', AdminAddPlanView.as_view(), name='admin_add_plan'),
    path('admin/add_payment_plan/', AddPaymentPlanView.as_view(), name='add_payment_plan'),
    path('admin/view_payment_plans/', ViewPaymentPlansView.as_view(), name='view_payment_plans'),
    path('admin/edit_payment_plan/<str:plan_id>/', EditPaymentPlanView.as_view(), name='edit_payment_plan'),
    path('admin/edit-payment-plan_page/<str:plan_id>/',EditPaymentPlanPageView.as_view(), name='edit_payment_plan_page'),
    
]
