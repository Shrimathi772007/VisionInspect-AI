import { useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { Eye, EyeOff, Mail, Lock, ScanLine, ShieldCheck, Layers, Camera } from "lucide-react";
import { useAuth } from "../auth/useAuth";
import { Input } from "../components/Input/Input";
import { Button } from "../components/Button/Button";
import { ApiError } from "../api/client";
import styles from "./LoginPage.module.css";

export function LoginPage() {
  const { login, isAuthenticated, isBootstrapping } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  if (!isBootstrapping && isAuthenticated) {
    const redirectTo = location.state?.from?.pathname || "/dashboard";
    return <Navigate to={redirectTo} replace />;
  }

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      await login(email, password);
      const redirectTo = location.state?.from?.pathname || "/dashboard";
      navigate(redirectTo, { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.visualPanel}>
        <div className={styles.grid} aria-hidden="true" />
        <div className={styles.glow} aria-hidden="true" />

        <div className={styles.visualContent}>
          <div className={styles.brand}>
            <div className={styles.brandMark}>
              <ScanLine size={20} strokeWidth={2} />
            </div>
            <span className={styles.brandName}>VisionInspect AI</span>
          </div>

          <h1 className={styles.headline}>
            Precision quality inspection, <span className={styles.headlineAccent}>powered by vision AI</span>
          </h1>
          <p className={styles.subheadline}>
            Capture, organize, and review manufacturing inspection imagery from a single, secure workspace built
            for quality engineering teams.
          </p>

          <ul className={styles.featureList}>
            <li>
              <Camera size={16} aria-hidden="true" />
              <span>Structured image acquisition per product line</span>
            </li>
            <li>
              <Layers size={16} aria-hidden="true" />
              <span>Centralized inspection records &amp; traceability</span>
            </li>
            <li>
              <ShieldCheck size={16} aria-hidden="true" />
              <span>Role-based access for engineers and supervisors</span>
            </li>
          </ul>
        </div>
      </div>

      <div className={styles.formPanel}>
        <div className={styles.formCard}>
          <div className={styles.formCardMobileBrand}>
            <div className={styles.brandMark}>
              <ScanLine size={18} strokeWidth={2} />
            </div>
            <span className={styles.brandName}>VisionInspect AI</span>
          </div>

          <h2 className={styles.formTitle}>Welcome back</h2>
          <p className={styles.formSubtitle}>Sign in to access the quality inspection workspace.</p>

          {error && (
            <div className={styles.errorBanner} role="alert">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className={styles.form} noValidate>
            <Input
              label="Email address"
              type="email"
              autoComplete="email"
              placeholder="you@company.com"
              leftIcon={<Mail size={16} />}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            <Input
              label="Password"
              type={showPassword ? "text" : "password"}
              autoComplete="current-password"
              placeholder="Enter your password"
              leftIcon={<Lock size={16} />}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              rightSlot={
                <button
                  type="button"
                  className={styles.togglePassword}
                  onClick={() => setShowPassword((v) => !v)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              }
            />

            <Button type="submit" size="lg" fullWidth loading={isSubmitting}>
              Sign in
            </Button>
          </form>

          <p className={styles.footerNote}>
            Access is provisioned internally. Contact your quality lead if you need an account.
          </p>
        </div>
      </div>
    </div>
  );
}
