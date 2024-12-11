from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views import View
from django.views.generic import ListView, CreateView, UpdateView
from extractionapp.models import PaymentPlan, YouTubeUser, Subscription_Data, Payment

# Admin Dashboard view: Lists all users, payments, and subscriptions
class AdminDashboardView(View):
    def get(self, request):
        users = YouTubeUser.objects.all()  # Fetch all users
        payments = Payment.objects.all()  # Fetch payments for all users
        subscriptions = Subscription_Data.objects.all()  # Fetch subscriptions for all users
        return render(request, 'adminapp/admin.html', {'users': users, 'payments': payments, 'subscriptions': subscriptions})

# Admin User Data view: Displays user data with payments and subscriptions
class AdminUserDataView(View):
    def get(self, request):
        users = YouTubeUser.objects.all()  # Fetch all users
        payments = Payment.objects.all()  # Fetch all payments
        subscriptions = Subscription_Data.objects.all()  # Fetch all subscriptions
        return render(request, 'adminapp/admin_user_data.html', {'users': users, 'payments': payments, 'subscriptions': subscriptions})

# Add Payment Plan view: Displays the form for adding a new payment plan
class AdminAddPlanView(View):
    def get(self, request):
        return render(request, 'adminapp/admin_add_plan.html')

# Add Payment Plan functionality: Handles form submission to create a new payment plan
class AddPaymentPlanView(View):
    def post(self, request):
        try:
            # Retrieve data from the POST request
            name = request.POST.get("name")
            plan_id = request.POST.get("plan_id")
            amount = request.POST.get("amount")
            interval = request.POST.get("interval")
            currency = request.POST.get("currency")

            # Validate that all necessary fields are present
            if not all([name, plan_id, amount, interval, currency]):
                return JsonResponse({"success": False, "error": "All fields are required."})

            # Save the data in the PaymentPlan model
            payment_plan = PaymentPlan.objects.create(
                name=name,
                plan_id=plan_id,
                amount=amount,
                interval=interval,
                currency=currency
            )

            # Alert the user
            return JsonResponse({"success": True, "message": "Payment plan added successfully."})

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})

# View all Payment Plans: List all existing payment plans
class ViewPaymentPlansView(ListView):
    model = PaymentPlan
    template_name = 'adminapp/view_payment_plans.html'
    context_object_name = 'payment_plans'

# Edit Payment Plan view: Allows editing of an existing payment plan
class EditPaymentPlanView(UpdateView):
    model = PaymentPlan
    template_name = 'adminapp/edit_payment_plan.html'
    context_object_name = 'plan'
    fields = ['name', 'amount', 'interval', 'currency']

    def form_valid(self, form):
        plan = form.save(commit=False)
        plan.save()
        # Return a success message after saving the plan
        return JsonResponse({"success": True, "message": "Payment plan updated successfully."})
    
class EditPaymentPlanPageView(UpdateView):
    model = PaymentPlan
    fields = ['name', 'amount', 'interval', 'currency']  # Fields you want to edit
    template_name = 'adminapp/edit_payment_plan.html'
    context_object_name = 'plan'
    success_url = '/admin/view_payment_plans/'  # Redirect after successful update

    def get_object(self, queryset=None):
        return get_object_or_404(PaymentPlan, plan_id=self.kwargs['plan_id'])    
