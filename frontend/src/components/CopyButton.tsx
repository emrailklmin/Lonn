import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { Button } from "./ui/button";
export function CopyButton({ value }: { value: string }) {
  const [done, setDone] = useState(false);
  return (
    <Button
      variant="outline"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        } catch {
          setDone(false);
        }
      }}
    >
      {done ? <Check size={15} /> : <Copy size={15} />}{" "}
      {done ? "Copied" : "Copy"}
    </Button>
  );
}
