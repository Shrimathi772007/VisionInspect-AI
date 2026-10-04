import { Navigate, Route, Routes } from "react-router-dom";
import { ProtectedRoute } from "./components/ProtectedRoute/ProtectedRoute";
import { RoleRoute } from "./components/RoleRoute/RoleRoute";
import { AppLayout } from "./layouts/AppLayout";
import { LoginPage } from "./pages/LoginPage";
import { RegisterPage } from "./pages/RegisterPage";
import { DashboardPage } from "./pages/DashboardPage";
import { ProductsPage } from "./pages/ProductsPage";
import { InspectionsPage } from "./pages/InspectionsPage";
import { InspectionUploadPage } from "./pages/InspectionUploadPage";
import { InspectionDetailPage } from "./pages/InspectionDetailPage";
import { DatasetBrowserPage } from "./pages/DatasetBrowserPage";
import { UsersPage } from "./pages/UsersPage";
import { ModelPerformancePage } from "./pages/ModelPerformancePage";
import { CameraSimulationPage } from "./pages/CameraSimulationPage";

function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      <Route
        element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        }
      >
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/products" element={<ProductsPage />} />
        <Route path="/inspections" element={<InspectionsPage />} />
        <Route
          path="/inspections/upload"
          element={
            <RoleRoute allow={["quality_engineer"]}>
              <InspectionUploadPage />
            </RoleRoute>
          }
        />
        <Route path="/inspections/:id" element={<InspectionDetailPage />} />
        <Route path="/dataset" element={<DatasetBrowserPage />} />
        <Route path="/models" element={<ModelPerformancePage />} />
        <Route
          path="/camera"
          element={
            <RoleRoute allow={["quality_engineer"]}>
              <CameraSimulationPage />
            </RoleRoute>
          }
        />
        <Route
          path="/users"
          element={
            <RoleRoute allow={["quality_engineer"]}>
              <UsersPage />
            </RoleRoute>
          }
        />
      </Route>

      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
}

export default App;
