import DropzoneUpload from "@/components/ui/dropzone-upload";
import ChipSelector from "@/components/ui/chip-selector";
import Button from "@/components/ui/button";
import { Trash2, Search } from "lucide-react";

const RESULT_OPTIONS = ["1", "5", "10"];

export default function ComposeSidebar({
  preview, onFileSelect, onRemove,
  numResults, onNumResultsChange,
  description, onDescriptionChange,
  onClear, onSubmit, loading,
}) {
  return (
    <div className="space-y-5">
      {/* Image upload */}
      <div>
        <p className="text-sm font-medium text-text-secondary mb-2">Reference Image</p>
        <DropzoneUpload preview={preview} onFileSelect={onFileSelect} onRemove={onRemove} />
      </div>

      {/* Text description */}
      <div>
        <p className="text-sm font-medium text-text-secondary mb-2">Modification Description</p>
        <textarea
          value={description}
          onChange={(e) => onDescriptionChange(e.target.value)}
          placeholder="e.g. same person but wearing a black jacket and jeans"
          className="w-full rounded-xl border border-border bg-white px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary/50 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary resize-none h-24"
        />
      </div>

      {/* Result count */}
      <ChipSelector
        label="Number of Results"
        options={RESULT_OPTIONS}
        value={String(numResults)}
        onChange={(v) => onNumResultsChange(Number(v))}
      />

      {/* Action buttons */}
      <div className="flex gap-2">
        <Button variant="outline" onClick={onClear} icon={<Trash2 className="h-4 w-4" />} className="flex-1">
          Clear
        </Button>
        <Button onClick={onSubmit} loading={loading} icon={<Search className="h-4 w-4" />} className="flex-1">
          Search
        </Button>
      </div>
    </div>
  );
}
