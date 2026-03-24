import { cn } from "@/lib/utils";
import Button from "@/components/ui/button";

export default function ComposeEvaluation({ index, value, onChange }) {
  return (
    <fieldset className="flex items-center gap-3 pt-1">
      <legend className="sr-only">Evaluate result {index + 1}</legend>
      {["True", "False"].map((option) => (
        <label
          key={option}
          className={cn(
            "flex items-center gap-1.5 text-xs font-medium cursor-pointer px-2 py-1 rounded-lg transition-colors",
            value === option
              ? option === "True" ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"
              : "text-text-secondary hover:bg-slate-50"
          )}
        >
          <input
            type="radio"
            name={`eval-${index}`}
            value={option}
            checked={value === option}
            onChange={() => onChange(option)}
            className="sr-only"
          />
          {option === "True" ? "Match" : "No Match"}
        </label>
      ))}
    </fieldset>
  );
}

export function EvaluationSubmitButton({ selectedMatches, resultsCount, onSubmit, evaluating }) {
  const allEvaluated = Object.keys(selectedMatches).length === resultsCount
    && !Object.values(selectedMatches).includes(undefined);

  return (
    <div className="mt-6 flex justify-center">
      <Button
        onClick={onSubmit}
        loading={evaluating}
        disabled={!allEvaluated}
        size="lg"
      >
        Submit Evaluations
      </Button>
    </div>
  );
}
