import { useState } from "react";
import { Search } from "lucide-react";
import PageTransition from "./layout/page-transition";
import LayoutContainer from "./layout/layout-container";
import SplitLayout from "./layout/split-layout";
import ComposeSidebar from "./compose/sidebar-controls";
import ComposeEvaluation, { EvaluationSubmitButton } from "./compose/compose-evaluation";
import ImageCard from "./ui/image-card";
import EmptyState from "./ui/empty-state";
import Skeleton from "./ui/skeleton";
import Toast from "./ui/toast";
import useComposeSearch from "@/shared/hooks/use-compose-search";
import useAuth from "@/shared/hooks/use-auth";

export default function ComposePage() {
  const { user } = useAuth();
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [numResults, setNumResults] = useState(10);
  const [description, setDescription] = useState("");
  const [selectedMatches, setSelectedMatches] = useState({});
  const [toast, setToast] = useState({ open: false, type: "success", title: "", message: "" });

  const { results, loading, error, evaluating, evalError, search, submitEvaluation, reset } =
    useComposeSearch();

  const handleFileSelect = (file) => {
    if (image) URL.revokeObjectURL(image);
    setImage(URL.createObjectURL(file));
    setImageFile(file);
    setSelectedMatches({});
  };

  const handleRemove = () => {
    if (image) URL.revokeObjectURL(image);
    setImage(null);
    setImageFile(null);
    reset();
    setSelectedMatches({});
  };

  const handleSubmit = async () => {
    if (!imageFile) return;
    setSelectedMatches({});
    try {
      await search(imageFile, { topK: numResults, description });
    } catch (err) {
      console.error("Compose search failed:", err);
    }
  };

  const handleEvaluationSubmit = async () => {
    try {
      const result = await submitEvaluation({
        selectedMatches,
        evaluatorCode: user?.username || "anonymous",
        description,
      });
      // Show success toast with accuracy
      const trueCount = result.true_count ?? 0;
      const total = result.total ?? 0;
      setToast({
        open: true,
        type: "success",
        title: "Evaluation submitted!",
        message: `Accuracy: ${trueCount}/${total} marked as True.`,
      });
    } catch (err) {
      console.error("Evaluation failed:", err);
      setToast({
        open: true,
        type: "error",
        title: "Evaluation failed",
        message: err.message || "Please try again.",
      });
    }
  };

  const sidebar = (
    <ComposeSidebar
      preview={image}
      onFileSelect={handleFileSelect}
      onRemove={handleRemove}
      numResults={numResults}
      onNumResultsChange={setNumResults}
      description={description}
      onDescriptionChange={setDescription}
      onClear={handleRemove}
      onSubmit={handleSubmit}
      loading={loading}
    />
  );

  return (
    <PageTransition>
      <LayoutContainer>
        <h1 className="text-2xl font-bold text-text-primary mb-6">Compose Search</h1>

        {error && <p className="text-sm text-error mb-4">{error}</p>}
        {evalError && <p className="text-sm text-error mb-4">{evalError}</p>}

        <SplitLayout sidebar={sidebar} sidebarTitle="Search Options">
          {results.length > 0 && (
            <p className="text-sm text-text-secondary mb-4">{results.length} result(s) found</p>
          )}

          {loading && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
              {Array.from({ length: numResults }).map((_, i) => (
                <Skeleton key={i} className="aspect-[3/4] rounded-xl" />
              ))}
            </div>
          )}

          {!loading && results.length === 0 && (
            <EmptyState
              icon={Search}
              title="No results yet"
              description="Upload a reference image, describe the changes, and click Search."
            />
          )}

          {!loading && results.length > 0 && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                {results.map((src, i) => (
                  <ImageCard key={i} src={src} index={i} alt={`Result ${i + 1}`}>
                    <ComposeEvaluation
                      index={i}
                      value={selectedMatches[i]}
                      onChange={(val) => setSelectedMatches((prev) => ({ ...prev, [i]: val }))}
                    />
                  </ImageCard>
                ))}
              </div>

              <EvaluationSubmitButton
                selectedMatches={selectedMatches}
                resultsCount={results.length}
                onSubmit={handleEvaluationSubmit}
                evaluating={evaluating}
              />
            </>
          )}
        </SplitLayout>
      </LayoutContainer>

      <Toast
        open={toast.open}
        type={toast.type}
        title={toast.title}
        message={toast.message}
        duration={5000}
        onClose={() => setToast((t) => ({ ...t, open: false }))}
      />
    </PageTransition>
  );
}
