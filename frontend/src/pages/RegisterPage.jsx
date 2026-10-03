import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { Eye, EyeOff, Mail, Lock, User, ScanLine, Info } from "lucide-react";
import { useAuth } from "../auth/useAuth";
import { registerUser } from "../api/auth";
import { ApiError } from "../api/client";
import { useToast } from "../components/Toast/ToastProvider";
import { Input } from "../components/Input/Input";
import { Button } from "../components/Button/Button";
import { ThemeToggle } from "../components/ThemeToggle/ThemeToggle";
import authStyles from "./LoginPage.module.css";
import styles from "./RegisterPage.module.css";

// Mirrors the backend's UserCreate schema (name 1-255, password 8-72 characters).
const NAME_MAX_LENGTH = 255;
const PASSWORD_MIN_LENGTH = 8;
const PASSWORD_MAX_LENGTH = 72;
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const EMPTY_FORM = { name: "", email: "", password: "", confirmPassword: "" };

function validate(values) {
  const errors = {};
  const name = values.name.trim();
  if (!name) errors.name = "Full name is required.";
  else if (name.length > NAME_MAX_LENGTH) errors.name = `Full name must be at most ${NAME_MAX_LENGTH} characters.`;

  if (!EMAIL_PATTERN.test(values.email.trim())) errors.email = "Enter a valid email address.";

  // Count characters (code points), as the server does, not UTF-16 units.
  const passwordLength = [...values.password].length;
  if (passwordLength < PASSWORD_MIN_LENGTH || passwordLength > PASSWORD_MAX_LENGTH) {
    errors.password = `Password must be between ${PASSWORD_MIN_LENGTH} and ${PASSWORD_MAX_LENGTH} characters.`;
  }

  if (values.confirmPassword !== values.password) errors.confirmPassword = "Passwords do not match.";
  return errors;
}

function registrationErrorMessage(err) {
  if (err instanceof ApiError && err.status === 409) return "An account with this email already exists.";
  // 422 (validation) and 0 (network) carry a message already normalized by the API client;
  // anything else (e.g. a 5xx) gets a generic message rather than raw server details.
  if (err instanceof ApiError && (err.status === 422 || err.status === 0)) return err.message;
  return "Something went wrong. Please try again.";
}

export function RegisterPage() {
  const { isAuthenticated, isBootstrapping } = useAuth();
  const { showToast } = useToast();
  const navigate = useNavigate();

  const [values, setValues] = useState(EMPTY_FORM);
  const [fieldErrors, setFieldErrors] = useState({});
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  if (!isBootstrapping && isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  const setField = (field) => (event) => setValues((current) => ({ ...current, [field]: event.target.value }));

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(null);
    const errors = validate(values);
    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) return;

    setIsSubmitting(true);
    try {
      await registerUser({ name: values.name.trim(), email: values.email.trim(), password: values.password });
      showToast({
        type: "success",
        title: "Account created",
        message: "Sign in with your new account.",
      });
      navigate("/login", { replace: true });
    } catch (err) {
      setError(registrationErrorMessage(err));
      setIsSubmitting(false);
    }
  };

  const passwordToggle = (
    <button
      type="button"
      className={authStyles.togglePassword}
      onClick={() => setShowPassword((v) => !v)}
      aria-label={showPassword ? "Hide password" : "Show password"}
    >
      {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
    </button>
  );

  return (
    <div className={authStyles.page}>
      <div className={authStyles.themeToggleWrap}>
        <ThemeToggle />
      </div>
      <div className={authStyles.visualPanel}>
        <div className={authStyles.grid} aria-hidden="true" />
        <div className={authStyles.glow} aria-hidden="true" />

        <div className={authStyles.visualContent}>
          <div className={authStyles.brand}>
            <div className={authStyles.brandMark}>
              <ScanLine size={20} strokeWidth={2} />
            </div>
            <span className={authStyles.brandName}>VisionInspect AI</span>
          </div>

          <h1 className={authStyles.headline}>
            Join the inspection workspace, <span className={authStyles.headlineAccent}>review with confidence</span>
          </h1>
          <p className={authStyles.subheadline}>
            Factory supervisors can review products, inspections, analytics and quality reports. Quality engineers
            manage inspections and user roles.
          </p>
        </div>
      </div>

      <div className={authStyles.formPanel}>
        <div className={authStyles.formCard}>
          <div className={authStyles.formCardMobileBrand}>
            <div className={authStyles.brandMark}>
              <ScanLine size={18} strokeWidth={2} />
            </div>
            <span className={authStyles.brandName}>VisionInspect AI</span>
          </div>

          <h2 className={authStyles.formTitle}>Create an account</h2>
          <p className={authStyles.formSubtitle}>Register to access the quality inspection workspace.</p>

          <div className={styles.roleNote}>
            <Info size={15} aria-hidden="true" />
            <span>New accounts are created as Factory Supervisor. A Quality Engineer can change your role.</span>
          </div>

          {error && (
            <div className={authStyles.errorBanner} role="alert">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className={authStyles.form} noValidate>
            <Input
              label="Full name"
              autoComplete="name"
              placeholder="Jane Doe"
              leftIcon={<User size={16} />}
              value={values.name}
              onChange={setField("name")}
              error={fieldErrors.name}
              required
            />
            <Input
              label="Email address"
              type="email"
              autoComplete="email"
              placeholder="you@company.com"
              leftIcon={<Mail size={16} />}
              value={values.email}
              onChange={setField("email")}
              error={fieldErrors.email}
              required
            />
            <Input
              label="Password"
              type={showPassword ? "text" : "password"}
              autoComplete="new-password"
              placeholder="8-72 characters"
              leftIcon={<Lock size={16} />}
              value={values.password}
              onChange={setField("password")}
              error={fieldErrors.password}
              required
              rightSlot={passwordToggle}
            />
            <Input
              label="Confirm password"
              type={showPassword ? "text" : "password"}
              autoComplete="new-password"
              placeholder="Re-enter your password"
              leftIcon={<Lock size={16} />}
              value={values.confirmPassword}
              onChange={setField("confirmPassword")}
              error={fieldErrors.confirmPassword}
              required
            />

            <Button type="submit" size="lg" fullWidth loading={isSubmitting}>
              Create account
            </Button>
          </form>

          <p className={authStyles.footerNote}>
            Already have an account?{" "}
            <Link to="/login" className={authStyles.footerLink}>
              Back to sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
