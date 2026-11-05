export function PythonCode({ code }: { code: string }) {
  const parts = code.split(
    /(#[^\n]*|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b(?:from|import|with|as|for|in|if|not|raise|print|try|except|None)\b|\b\d+(?:\.\d+)?\b)/g,
  );
  return (
    <>
      {parts.map((part, i) => (
        <span
          key={i}
          className={
            part.startsWith("#")
              ? "syntax-comment"
              : part.startsWith('"') || part.startsWith("'")
                ? "syntax-string"
                : /^(from|import|with|as|for|in|if|not|raise|try|except|None)$/.test(
                      part,
                    )
                  ? "syntax-keyword"
                  : /^\d/.test(part)
                    ? "syntax-number"
                    : ""
          }
        >
          {part}
        </span>
      ))}
    </>
  );
}
