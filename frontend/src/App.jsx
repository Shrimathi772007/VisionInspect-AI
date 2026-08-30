import { Navigate, Route, Routes } from "react-router-dom";
import { ProtectedRoute } from "./components/ProtectedRoute/ProtectedRoute";
import { AppLayout } from "./layouts/AppLayout";
import { LoginPage } from "./pages/LoginPage";
import { DashboardPage } from "./pages/DashboardPage";
import { ProductsPage } from "./pages/ProductsPage";
import { InspectionsPage } from "./pages/InspectionsPage";
import { InspectionUploadPage } from "./pages/InspectionUploadPage";
import { InspectionDetailPage } from "./pages/InspectionDetailPage";
import { DatasetBrowserPage } from "./pages/DatasetBrowserPage";

function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />

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
        <Route path="/inspections/upload" element={<InspectionUploadPage />} />
        <Route path="/inspections/:id" element={<InspectionDetailPage />} />
        <Route path="/dataset" element={<DatasetBrowserPage />} />
      </Route>

      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
}

export default App;
