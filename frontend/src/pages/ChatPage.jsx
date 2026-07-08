import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import api from "../api";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

/** Open links safely in a new tab; react-markdown never emits raw HTML by default. */
const MARKDOWN_COMPONENTS = {
  a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
};

function AssistantMarkdown({ content }) {
  return (
    <div className="bubble bubble--markdown markdown-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
        {String(content || "")}
      </ReactMarkdown>
    </div>
  );
}

function isStreamPlaceholderMessage(message) {
  return String(message?.id ?? "").startsWith("stream-");
}

function DocIconUpload({ className = "", ...rest }) {
  return (
    <svg
      className={["doc-action-icon", className].filter(Boolean).join(" ")}
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function DocIconView({ className = "", ...rest }) {
  return (
    <svg
      className={["doc-action-icon", "doc-action-icon--view", className].filter(Boolean).join(" ")}
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.85"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="8" y1="13" x2="16" y2="13" />
      <line x1="8" y1="17" x2="14" y2="17" />
      <line x1="8" y1="9" x2="12" y2="9" />
    </svg>
  );
}

function DocIconTrash({ className = "", ...rest }) {
  return (
    <svg
      className={["doc-action-icon", className].filter(Boolean).join(" ")}
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}

function SendPromptIcon({ className = "", ...rest }) {
  return (
    <svg
      className={className}
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.25"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      <path d="M12 19V7M8 11l4-4 4 4" />
    </svg>
  );
}

function IconClose({ className = "", ...rest }) {
  return (
    <svg
      className={className}
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function IconCopy({ className = "", ...rest }) {
  return (
    <svg
      className={className}
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function IconStop({ className = "", ...rest }) {
  return (
    <svg
      className={className}
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden
      {...rest}
    >
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}

function IconThumbUp({ className = "", ...rest }) {
  return (
    <svg
      className={className}
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      <path d="M7 10v11" />
      <path d="M17 10h-4l1-5v-1a2 2 0 0 0-2-2l-3 7v11h9a2 2 0 0 0 2-2l1-6.5a2 2 0 0 0-2-2.5z" />
    </svg>
  );
}

function IconThumbDown({ className = "", ...rest }) {
  return (
    <svg
      className={className}
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      <path d="M17 14V3" />
      <path d="M7 14h4l-1 5v1a2 2 0 0 0 2 2l3-7V4H6a2 2 0 0 0-2 2l-1 6.5A2 2 0 0 0 5 15h2z" />
    </svg>
  );
}

function DocIconEdit({ className = "", ...rest }) {
  return (
    <svg
      className={["doc-action-icon", "doc-action-icon--edit", className].filter(Boolean).join(" ")}
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
    </svg>
  );
}

function IconMoreHorizontal({ className = "", ...rest }) {
  return (
    <svg
      className={className}
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden
      {...rest}
    >
      <circle cx="5" cy="12" r="2" />
      <circle cx="12" cy="12" r="2" />
      <circle cx="19" cy="12" r="2" />
    </svg>
  );
}

function IconPin({ className = "", ...rest }) {
  return (
    <svg
      className={className}
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      <line x1="12" x2="12" y1="17" y2="22" />
      <path d="M5 17h14v-1a7 7 0 0 0-7-7 7 7 0 0 0-7 7v1Z" />
    </svg>
  );
}

function IconArchive({ className = "", ...rest }) {
  return (
    <svg
      className={className}
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      <rect x="3" y="4" width="18" height="4" rx="1" />
      <path d="M5 8v11a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8" />
      <path d="M10 12h4" />
    </svg>
  );
}

function isChatPinned(chat) {
  return chat?.pinned_at != null && chat.pinned_at !== "";
}

/** Coerce DB / JSON types so each assistant row can show retrieval & accuracy after reload. */
function normalizeMessageRow(m) {
  if (!m || typeof m !== "object") return m;
  const out = { ...m };
  if (out.retrieval_seconds != null && out.retrieval_seconds !== "") {
    const n = Number(out.retrieval_seconds);
    if (Number.isFinite(n)) out.retrieval_seconds = n;
    else delete out.retrieval_seconds;
  } else {
    delete out.retrieval_seconds;
  }
  if (out.latency_seconds != null && out.latency_seconds !== "") {
    const n = Number(out.latency_seconds);
    if (Number.isFinite(n)) out.latency_seconds = n;
    else delete out.latency_seconds;
  } else {
    delete out.latency_seconds;
  }
  if (out.accuracy_score != null && out.accuracy_score !== "") {
    const n = Number(out.accuracy_score);
    if (Number.isFinite(n)) out.accuracy_score = n;
    else delete out.accuracy_score;
  } else {
    delete out.accuracy_score;
  }
  if (!out.accuracy_label) delete out.accuracy_label;
  if (Array.isArray(out.retrieval_context)) {
    out.retrieval_context = out.retrieval_context
      .map((c) => String(c ?? "").trim())
      .filter((c) => c.length > 0);
  } else if (out.retrieval_context != null && typeof out.retrieval_context === "string") {
    try {
      const parsed = JSON.parse(out.retrieval_context);
      out.retrieval_context = Array.isArray(parsed)
        ? parsed.map((c) => String(c ?? "").trim()).filter((c) => c.length > 0)
        : [];
    } catch {
      out.retrieval_context = [];
    }
  } else {
    out.retrieval_context = [];
  }
  if (Array.isArray(out.source_citations)) {
    out.source_citations = out.source_citations.filter((c) => c && typeof c === "object");
  } else if (out.source_citations != null && typeof out.source_citations === "string") {
    try {
      const parsed = JSON.parse(out.source_citations);
      out.source_citations = Array.isArray(parsed) ? parsed.filter((c) => c && typeof c === "object") : [];
    } catch {
      out.source_citations = [];
    }
  } else {
    out.source_citations = [];
  }
  if (out.feedback_rating != null && out.feedback_rating !== "") {
    const n = Number(out.feedback_rating);
    out.feedback_rating = Number.isFinite(n) ? n : 0;
  } else {
    out.feedback_rating = 0;
  }
  return out;
}

function normalizeMessagesFromApi(rows) {
  return (rows || []).map(normalizeMessageRow);
}

export default function ChatPage() {
  const navigate = useNavigate();
  const [files, setFiles] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [chats, setChats] = useState([]);
  const [selectedChatId, setSelectedChatId] = useState(null);
  const [isCreatingNewChat, setIsCreatingNewChat] = useState(false);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [deletingDocument, setDeletingDocument] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [toastMessage, setToastMessage] = useState("");
  const [exportFormat, setExportFormat] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const [showScrollToLatest, setShowScrollToLatest] = useState(false);
  const isNearBottomRef = useRef(true);
  const [editingUserMessageId, setEditingUserMessageId] = useState(null);
  const [editingDraft, setEditingDraft] = useState("");
  const [editingSidebarChatId, setEditingSidebarChatId] = useState(null);
  const [editingSidebarTitle, setEditingSidebarTitle] = useState("");
  const [renamingChat, setRenamingChat] = useState(false);
  const [deleteChatTarget, setDeleteChatTarget] = useState(null);
  const [deletingChat, setDeletingChat] = useState(false);
  const [chatMenu, setChatMenu] = useState(null);
  const selectedChatIdRef = useRef(null);
  const startFreshRef = useRef(localStorage.getItem("startFreshChat") === "1");
  const messagesContainerRef = useRef(null);
  const fileListRef = useRef(null);
  const chatItemRefs = useRef({});
  const scrollAnimationRef = useRef(null);
  const scrollTargetRef = useRef(0);
  const isScrollAnimatingRef = useRef(false);
  const sidebarScrollAnimationRef = useRef(null);
  const sidebarScrollTargetRef = useRef(0);
  const isSidebarScrollAnimatingRef = useRef(false);
  const isUserScrollingSidebarRef = useRef(false);
  const sidebarScrollIdleTimerRef = useRef(null);
  const questionInputRef = useRef(null);
  const renameChatInputRef = useRef(null);
  const userMessageEditInputRef = useRef(null);
  const chatOverflowMenuRef = useRef(null);
  const messagesEndRef = useRef(null);
  const toastTimerRef = useRef(null);
  const streamAbortRef = useRef(null);
  /** Fallback merge when a row still lacks metrics (e.g. old rows) after reload. */
  const lastStreamAssistantMetaRef = useRef(null);
  const [showPdfViewer, setShowPdfViewer] = useState(false);
  const [pdfObjectUrl, setPdfObjectUrl] = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const pdfObjectUrlRef = useRef(null);
  const user = useMemo(() => JSON.parse(localStorage.getItem("user") || "{}"), []);

  useEffect(() => {
    selectedChatIdRef.current = selectedChatId;
  }, [selectedChatId]);

  useEffect(() => {
    if (editingSidebarChatId == null) return;
    requestAnimationFrame(() => {
      renameChatInputRef.current?.focus();
      renameChatInputRef.current?.select();
    });
  }, [editingSidebarChatId]);

  useLayoutEffect(() => {
    if (editingUserMessageId == null) return;
    const el = userMessageEditInputRef.current;
    if (!el) return;
    const syncScroll = () => {
      el.scrollTop = 0;
      el.scrollLeft = 0;
    };
    syncScroll();
    requestAnimationFrame(syncScroll);
  }, [editingUserMessageId]);

  useEffect(() => {
    setEditingSidebarChatId(null);
    setEditingSidebarTitle("");
    setDeleteChatTarget(null);
    setChatMenu(null);
  }, [selectedDocument?.id]);

  useEffect(() => {
    if (!chatMenu) return;
    const onDocMouseDown = (e) => {
      if (chatOverflowMenuRef.current?.contains(e.target)) return;
      if (e.target.closest?.("[data-chat-overflow-trigger]")) return;
      setChatMenu(null);
    };
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [chatMenu]);

  useEffect(() => {
    if (!chatMenu) return;
    const close = () => setChatMenu(null);
    const node = fileListRef.current;
    node?.addEventListener("scroll", close, { passive: true });
    window.addEventListener("resize", close);
    return () => {
      node?.removeEventListener("scroll", close);
      window.removeEventListener("resize", close);
    };
  }, [chatMenu]);

  useEffect(() => {
    if (!chatMenu) return;
    const onKey = (e) => {
      if (e.key === "Escape") setChatMenu(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chatMenu]);

  const animateScrollToBottom = (smooth = true) => {
    const node = messagesContainerRef.current;
    if (!node) return;
    scrollTargetRef.current = Math.max(0, node.scrollHeight - node.clientHeight);
    if (!smooth) {
      node.scrollTop = scrollTargetRef.current;
      return;
    }
    if (isScrollAnimatingRef.current) return;

    isScrollAnimatingRef.current = true;
    const step = () => {
      const currentNode = messagesContainerRef.current;
      if (!currentNode) {
        isScrollAnimatingRef.current = false;
        return;
      }
      const target = scrollTargetRef.current;
      const distance = target - currentNode.scrollTop;

      if (Math.abs(distance) < 1) {
        currentNode.scrollTop = target;
        isScrollAnimatingRef.current = false;
        scrollAnimationRef.current = null;
        return;
      }

      currentNode.scrollTop += distance * 0.22;
      scrollAnimationRef.current = requestAnimationFrame(step);
    };

    scrollAnimationRef.current = requestAnimationFrame(step);
  };

  const animateSidebarToActive = () => {
    const listNode = fileListRef.current;
    const activeNode = selectedChatId ? chatItemRefs.current[selectedChatId] : null;
    if (!listNode || !activeNode) return;
    if (isUserScrollingSidebarRef.current) return;

    const target = Math.max(
      0,
      activeNode.offsetTop - listNode.clientHeight / 2 + activeNode.clientHeight / 2
    );
    sidebarScrollTargetRef.current = target;
    if (isSidebarScrollAnimatingRef.current) return;

    isSidebarScrollAnimatingRef.current = true;
    const step = () => {
      const node = fileListRef.current;
      if (!node) {
        isSidebarScrollAnimatingRef.current = false;
        return;
      }

      const distance = sidebarScrollTargetRef.current - node.scrollTop;
      if (Math.abs(distance) < 1) {
        node.scrollTop = sidebarScrollTargetRef.current;
        isSidebarScrollAnimatingRef.current = false;
        sidebarScrollAnimationRef.current = null;
        return;
      }

      node.scrollTop += distance * 0.2;
      sidebarScrollAnimationRef.current = requestAnimationFrame(step);
    };

    sidebarScrollAnimationRef.current = requestAnimationFrame(step);
  };

  const loadFiles = async (preferredDocumentId = null) => {
    try {
      const { data } = await api.get("/files");
      setFiles(data.files || []);
      const nextSelected =
        data.files?.find((file) => file.id === preferredDocumentId) ||
        data.files?.find((file) => file.id === selectedDocument?.id) ||
        data.files?.[0] ||
        null;
      setSelectedDocument(nextSelected);
    } catch (err) {
      if (err.response?.status === 401) {
        localStorage.clear();
        navigate("/login");
      }
    }
  };

  const loadChatListOnly = async (documentId) => {
    if (!documentId) return;
    try {
      const { data } = await api.get("/chats", { params: { document_id: documentId } });
      setChats(data.chats || []);
    } catch (err) {
      if (err.response?.status === 401) {
        localStorage.clear();
        navigate("/login");
      }
    }
  };

  const loadChats = async (documentId, forceChatId = null, keepFreshComposer = false) => {
    if (!documentId) {
      setChats([]);
      setSelectedChatId(null);
      selectedChatIdRef.current = null;
      setMessages([]);
      return;
    }
    try {
      const { data } = await api.get("/chats", { params: { document_id: documentId } });
      const nextChats = data.chats || [];
      setChats(nextChats);

      const currentChatId = selectedChatIdRef.current;
      const shouldKeepFreshComposer = forceChatId === null && (isCreatingNewChat || keepFreshComposer);
      const resolvedChatId = shouldKeepFreshComposer
        ? null
        : forceChatId ?? (nextChats.some((chat) => chat.id === currentChatId) ? currentChatId : nextChats[0]?.id ?? null);

      setSelectedChatId(resolvedChatId);
      selectedChatIdRef.current = resolvedChatId;

      if (resolvedChatId) {
        const messagesResp = await api.get(`/chats/${resolvedChatId}/messages`);
        setMessages(
          normalizeMessagesFromApi(
            (messagesResp.data.messages || []).filter((row) => String(row?.content || "").trim().length > 0)
          )
        );
        setIsCreatingNewChat(false);
      } else {
        setMessages([]);
      }
    } catch (err) {
      if (err.response?.status === 401) {
        localStorage.clear();
        navigate("/login");
      } else {
        setError(err.response?.data?.detail || "Failed to load chat history.");
      }
    }
  };

  const calculateAccuracyFromDistances = (distances = []) => {
    if (!Array.isArray(distances) || !distances.length) {
      return { label: "Low", score: 0 };
    }
    const avgDistance = distances.reduce((sum, value) => sum + Number(value || 0), 0) / distances.length;
    const score = Math.max(0, Math.min(100, Math.round((1.2 - avgDistance) * 100)));
    if (score >= 80) return { label: "High", score };
    if (score >= 55) return { label: "Medium", score };
    return { label: "Low", score };
  };

  const formatAssistantMetrics = (message) => {
    const parts = [];
    if (Number.isFinite(message.retrieval_seconds) && message.retrieval_seconds >= 0) {
      parts.push(`Retrieval: ${Number(message.retrieval_seconds).toFixed(2)}s`);
    }
    if (message.accuracy_label) {
      if (message.accuracy_label === "N/A") parts.push("Accuracy: N/A");
      else if (Number.isFinite(message.accuracy_score)) {
        parts.push(`Accuracy: ${message.accuracy_label} (${message.accuracy_score}%)`);
      }
    }
    if (Number.isFinite(message.latency_seconds)) {
      parts.push(`Total: ${message.latency_seconds}s`);
    }
    if (Array.isArray(message.source_citations) && message.source_citations.length) {
      parts.push(`Sources: ${message.source_citations.length}`);
    }
    return parts.length ? parts.join(" • ") : "";
  };

  const submitAssistantFeedback = async (messageId, rating) => {
    if (!Number.isFinite(Number(messageId))) return;
    try {
      await api.post(`/messages/${messageId}/feedback`, { rating });
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId ? { ...msg, feedback_rating: rating } : msg
        )
      );
      showSuccessToast(rating > 0 ? "Marked helpful." : "Marked not helpful.");
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to save feedback.");
    }
  };

  useEffect(() => {
    loadFiles();
  }, []);

  useEffect(() => {
    const docId = selectedDocument?.id || null;
    if (!docId) {
      loadChats(null);
      return;
    }

    const shouldStartFresh = startFreshRef.current;
    if (shouldStartFresh) {
      setSelectedChatId(null);
      selectedChatIdRef.current = null;
      setMessages([]);
      setIsCreatingNewChat(true);
      localStorage.removeItem("startFreshChat");
      startFreshRef.current = false;
      loadChats(docId, null, true);
      return;
    }

    setIsCreatingNewChat(false);
    loadChats(docId);
  }, [selectedDocument?.id]);

  const handleMessagesScroll = () => {
    const node = messagesContainerRef.current;
    if (!node) return;
    const distanceFromBottom = node.scrollHeight - node.scrollTop - node.clientHeight;
    const near = distanceFromBottom < 120;
    isNearBottomRef.current = near;
    setShowScrollToLatest((prev) => (prev === !near ? prev : !near));
  };

  const jumpToLatest = () => {
    isNearBottomRef.current = true;
    setShowScrollToLatest(false);
    animateScrollToBottom(true);
  };

  useEffect(() => {
    // Only follow new content when the user is already reading the latest turn,
    // so scrolling up to re-read isn't yanked back down mid-stream.
    if (isNearBottomRef.current) animateScrollToBottom(true);
  }, [messages]);

  useEffect(() => {
    // Switching chats should always land on the newest message.
    isNearBottomRef.current = true;
    setShowScrollToLatest(false);
    animateScrollToBottom(false);
  }, [selectedChatId]);

  useEffect(() => {
    if (!isNearBottomRef.current) return;
    // Keep newest user question / streamed answer visible.
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, sending]);

  useEffect(() => {
    const el = questionInputRef.current;
    if (!el || el.tagName !== "TEXTAREA") return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [question]);

  useEffect(() => {
    animateSidebarToActive();
  }, [selectedChatId]);

  useEffect(() => {
    return () => {
      if (scrollAnimationRef.current) {
        cancelAnimationFrame(scrollAnimationRef.current);
      }
      if (sidebarScrollAnimationRef.current) {
        cancelAnimationFrame(sidebarScrollAnimationRef.current);
      }
      if (sidebarScrollIdleTimerRef.current) {
        clearTimeout(sidebarScrollIdleTimerRef.current);
      }
      if (toastTimerRef.current) {
        clearTimeout(toastTimerRef.current);
      }
    };
  }, []);

  const showSuccessToast = (message) => {
    setToastMessage(message);
    if (toastTimerRef.current) {
      clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = setTimeout(() => {
      setToastMessage("");
      toastTimerRef.current = null;
    }, 2600);
  };

  const uploadFile = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadProgress(0);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const { data } = await api.post("/files/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          if (!progressEvent.total) return;
          const percent = Math.min(100, Math.round((progressEvent.loaded * 100) / progressEvent.total));
          setUploadProgress(percent);
        },
      });
      setUploadProgress(100);
      await loadFiles(data?.document_id ?? null);
    } catch (err) {
      setError(err.response?.data?.error || err.response?.data?.detail || "Upload failed.");
    } finally {
      setUploading(false);
      setTimeout(() => setUploadProgress(0), 300);
      event.target.value = "";
    }
  };

  const reloadMessagesFromServer = async () => {
    const cid = selectedChatIdRef.current;
    if (!cid) return;
    try {
      const { data } = await api.get(`/chats/${cid}/messages`);
      let rows = normalizeMessagesFromApi(
        (data.messages || []).filter((m) => String(m?.content || "").trim().length > 0)
      );
      const merge = lastStreamAssistantMetaRef.current;
      if (merge && rows.length) {
        let lastAi = -1;
        for (let i = rows.length - 1; i >= 0; i--) {
          if (rows[i].role === "assistant") {
            lastAi = i;
            break;
          }
        }
        if (lastAi >= 0) {
          rows = rows.slice();
          const base = { ...rows[lastAi] };
          if (
            merge.retrieval_seconds != null &&
            Number.isFinite(Number(merge.retrieval_seconds)) &&
            base.retrieval_seconds == null
          ) {
            base.retrieval_seconds = Number(merge.retrieval_seconds);
          }
          if (Number.isFinite(merge.latency_seconds) && base.latency_seconds == null) {
            base.latency_seconds = merge.latency_seconds;
          }
          if (merge.accuracy_label != null && merge.accuracy_label !== "" && base.accuracy_label == null) {
            base.accuracy_label = merge.accuracy_label;
            base.accuracy_score = merge.accuracy_score;
          }
          rows[lastAi] = base;
        }
        lastStreamAssistantMetaRef.current = null;
      }
      setMessages(rows);
    } catch {
      /* ignore */
    }
  };

  const executeStreamRequest = async (messageText, streamMessageId, replaceUserMessageId) => {
    const startedAt = performance.now();
    let streamText = "";
    let activeChatId = selectedChatIdRef.current;
    let donePayload = null;
    try {
      const controller = new AbortController();
      streamAbortRef.current = controller;
      const token = localStorage.getItem("token");
      const body = {
        document_id: selectedDocument.id,
        question: messageText,
        chat_id: selectedChatIdRef.current ?? null,
      };
      if (replaceUserMessageId != null) {
        body.replace_user_message_id = replaceUserMessageId;
      }
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: "POST",
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          Authorization: token ? `Bearer ${token}` : "",
        },
        body: JSON.stringify(body),
      });

      if (response.status === 401) {
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        navigate("/login", { replace: true });
        throw new Error("Session expired. Please sign in again.");
      }

      if (!response.ok || !response.body) {
        throw new Error("Failed to start streaming response.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const rawEvent of events) {
          const dataLine = rawEvent
            .split("\n")
            .find((line) => line.startsWith("data: "));
          if (!dataLine) continue;
          const payload = JSON.parse(dataLine.slice(6));

          if (payload.type === "token") {
            streamText += payload.delta || "";
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === streamMessageId ? { ...msg, content: `${msg.content || ""}${payload.delta || ""}` } : msg
              )
            );
          } else if (payload.type === "meta") {
            if (payload.chat_id != null) {
              const cid = Number(payload.chat_id);
              if (Number.isFinite(cid)) {
                activeChatId = cid;
                selectedChatIdRef.current = cid;
                setSelectedChatId(cid);
                setIsCreatingNewChat(false);
              }
            }
            if (payload.retrieval_seconds != null) {
              const rs = Number(payload.retrieval_seconds);
              if (Number.isFinite(rs)) {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === streamMessageId ? { ...msg, retrieval_seconds: rs } : msg
                  )
                );
              }
            }
          } else if (payload.type === "done") {
            donePayload = payload;
            if (payload.chat_id) {
              activeChatId = payload.chat_id;
            }
            const elapsedSeconds = Number(((performance.now() - startedAt) / 1000).toFixed(2));
            const serverLat =
              payload.latency_seconds != null && Number.isFinite(Number(payload.latency_seconds))
                ? Number(payload.latency_seconds)
                : elapsedSeconds;
            const hasServerAccuracy =
              payload.accuracy_label !== undefined && payload.accuracy_score !== undefined;
            const accuracy = hasServerAccuracy
              ? { label: payload.accuracy_label, score: Number(payload.accuracy_score) }
              : calculateAccuracyFromDistances(payload.distances || []);
            const rs =
              payload.retrieval_seconds != null && Number.isFinite(Number(payload.retrieval_seconds))
                ? Number(payload.retrieval_seconds)
                : undefined;
            const mergedRs = rs !== undefined ? rs : undefined;
            lastStreamAssistantMetaRef.current = {
              retrieval_seconds: mergedRs,
              latency_seconds: serverLat,
              accuracy_label: accuracy.label,
              accuracy_score: accuracy.score,
            };
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === streamMessageId
                  ? {
                      ...msg,
                      content: payload.answer || msg.content,
                      latency_seconds: serverLat,
                      retrieval_seconds: mergedRs !== undefined ? mergedRs : msg.retrieval_seconds,
                      accuracy_label: accuracy.label,
                      accuracy_score: accuracy.score,
                      source_citations: Array.isArray(payload.source_citations) ? payload.source_citations : [],
                    }
                  : msg
              )
            );
          } else if (payload.type === "error") {
            throw new Error(payload.detail || "Streaming failed.");
          }
        }
      }

      if (donePayload?.chat_id) {
        if (!selectedChatIdRef.current || selectedChatIdRef.current !== donePayload.chat_id) {
          setSelectedChatId(donePayload.chat_id);
          selectedChatIdRef.current = donePayload.chat_id;
        }
        setIsCreatingNewChat(false);
        await loadChatListOnly(selectedDocument.id);
        await reloadMessagesFromServer();
      }
    } catch (err) {
      if (err?.name === "AbortError") {
        const stoppedContent = streamText.trim() || "Response stopped.";
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === streamMessageId ? { ...msg, content: stoppedContent, stopped: true } : msg
          )
        );
        if (activeChatId) {
          try {
            await api.post(`/chats/${activeChatId}/messages/assistant`, { content: stoppedContent });
            if (!selectedChatIdRef.current) {
              setSelectedChatId(activeChatId);
              selectedChatIdRef.current = activeChatId;
            }
            setIsCreatingNewChat(false);
            if (selectedDocument?.id) {
              await loadChatListOnly(selectedDocument.id);
            }
            await reloadMessagesFromServer();
          } catch (_saveErr) {
            // Non-fatal: user still sees stopped content in UI.
          }
        }
        showSuccessToast("Stopped");
      } else {
        if (replaceUserMessageId != null && selectedChatIdRef.current) {
          try {
            await reloadMessagesFromServer();
          } catch {
            /* ignore */
          }
        } else {
          setMessages((prev) => prev.filter((msg) => msg.id !== streamMessageId));
        }
        setError(err?.message || "Failed to generate response.");
      }
    } finally {
      streamAbortRef.current = null;
      setSending(false);
      requestAnimationFrame(() => {
        questionInputRef.current?.focus();
      });
    }
  };

  const sendQuestion = async (event) => {
    event.preventDefault();
    if (!selectedDocument || !question.trim() || sending) return;
    const userMessage = { role: "user", content: question.trim() };
    const streamMessageId = `stream-${Date.now()}`;
    isNearBottomRef.current = true;
    setShowScrollToLatest(false);
    setMessages((prev) => [...prev, userMessage, { id: streamMessageId, role: "assistant", content: "" }]);
    const messageText = question.trim();
    setQuestion("");
    setSending(true);
    setError("");
    await executeStreamRequest(messageText, streamMessageId, null);
  };

  const cancelEditUserMessage = () => {
    setEditingUserMessageId(null);
    setEditingDraft("");
  };

  const cancelRenameSidebarChat = () => {
    setEditingSidebarChatId(null);
    setEditingSidebarTitle("");
  };

  const beginRenameSidebarChat = (chat) => {
    if (sending || renamingChat || !chat?.id) return;
    setChatMenu(null);
    setDeleteChatTarget(null);
    setEditingSidebarChatId(chat.id);
    setEditingSidebarTitle(String(chat.title || "").trim() || "Conversation");
  };

  const submitRenameSidebarChat = async (event) => {
    event.preventDefault();
    const nextTitle = editingSidebarTitle.trim();
    if (!nextTitle || editingSidebarChatId == null || !selectedDocument?.id || renamingChat) return;
    setRenamingChat(true);
    setError("");
    try {
      await api.patch(`/chats/${editingSidebarChatId}`, { title: nextTitle });
      await loadChatListOnly(selectedDocument.id);
      showSuccessToast("Conversation renamed.");
      cancelRenameSidebarChat();
    } catch (err) {
      if (err.response?.status === 401) {
        localStorage.clear();
        navigate("/login");
      } else {
        setError(err.response?.data?.detail || "Could not rename this conversation.");
      }
    } finally {
      setRenamingChat(false);
    }
  };

  const confirmSoftDeleteChat = async () => {
    if (!deleteChatTarget?.id || !selectedDocument || deletingChat) return;
    const chatId = deleteChatTarget.id;
    const wasSelected = selectedChatIdRef.current === chatId;
    setDeletingChat(true);
    setError("");
    try {
      await api.delete(`/chats/${chatId}`);
      const { data } = await api.get("/chats", { params: { document_id: selectedDocument.id } });
      const nextChats = data.chats || [];
      setChats(nextChats);
      setDeleteChatTarget(null);
      showSuccessToast("Conversation removed.");
      if (wasSelected) {
        const nextId = nextChats[0]?.id ?? null;
        if (nextId) {
          await openExistingChat(nextId);
        } else {
          openNewChat();
        }
      }
    } catch (err) {
      if (err.response?.status === 401) {
        localStorage.clear();
        navigate("/login");
      } else {
        setError(err.response?.data?.detail || "Could not remove this conversation.");
      }
    } finally {
      setDeletingChat(false);
    }
  };

  const handleTogglePinChat = async (chat) => {
    if (!selectedDocument?.id || !chat?.id) return;
    setChatMenu(null);
    setError("");
    const nextPinned = !isChatPinned(chat);
    try {
      await api.patch(`/chats/${chat.id}`, { pinned: nextPinned });
      await loadChatListOnly(selectedDocument.id);
      showSuccessToast(nextPinned ? "Chat pinned." : "Chat unpinned.");
    } catch (err) {
      if (err.response?.status === 401) {
        localStorage.clear();
        navigate("/login");
      } else {
        setError(err.response?.data?.detail || "Could not update pin.");
      }
    }
  };

  const handleArchiveChat = async (chat) => {
    if (!selectedDocument?.id || !chat?.id) return;
    setChatMenu(null);
    const cid = chat.id;
    const wasSelected = selectedChatIdRef.current === cid;
    setError("");
    try {
      await api.patch(`/chats/${cid}`, { archived: true });
      showSuccessToast("Chat archived.");
      const { data } = await api.get("/chats", { params: { document_id: selectedDocument.id } });
      const nextChats = data.chats || [];
      setChats(nextChats);
      if (wasSelected) {
        const nextId = nextChats[0]?.id ?? null;
        if (nextId) {
          await openExistingChat(nextId);
        } else {
          openNewChat();
        }
      }
    } catch (err) {
      if (err.response?.status === 401) {
        localStorage.clear();
        navigate("/login");
      } else {
        setError(err.response?.data?.detail || "Could not archive this chat.");
      }
    }
  };

  const submitEditedUserMessage = async (messageId) => {
    const next = editingDraft.trim();
    if (!next || !selectedDocument || !selectedChatId || sending) return;
    const mid = Number(messageId);
    if (!Number.isFinite(mid)) return;
    const idx = messages.findIndex((m) => m.id === mid);
    if (idx === -1) return;
    cancelEditUserMessage();
    const streamMessageId = `stream-${Date.now()}`;
    isNearBottomRef.current = true;
    setShowScrollToLatest(false);
    setMessages((prev) => {
      const updated = prev.map((m, i) => (i === idx ? { ...m, content: next } : m));
      const head = updated.slice(0, idx + 1);
      const tail = updated.slice(idx + 1);
      const withoutFollowingAssistant = tail[0]?.role === "assistant" ? tail.slice(1) : tail;
      return [...head, { id: streamMessageId, role: "assistant", content: "" }, ...withoutFollowingAssistant];
    });
    setSending(true);
    setError("");
    await executeStreamRequest(next, streamMessageId, mid);
  };

  const stopProcess = () => {
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
    }
  };

  const openNewChat = () => {
    setSelectedChatId(null);
    selectedChatIdRef.current = null;
    setIsCreatingNewChat(true);
    setMessages([]);
    setQuestion("");
    setError("");
    cancelEditUserMessage();
    cancelRenameSidebarChat();
    setChatMenu(null);
  };

  const deleteSelectedDocument = async () => {
    if (!selectedDocument || deletingDocument) return;

    setDeletingDocument(true);
    setError("");
    try {
      await api.delete(`/files/${selectedDocument.id}`);
      await loadFiles();
      setShowDeleteModal(false);
      showSuccessToast("PDF deleted successfully.");
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to delete this document.");
    } finally {
      setDeletingDocument(false);
    }
  };

  const openExistingChat = async (chatId) => {
    setSelectedChatId(chatId);
    selectedChatIdRef.current = chatId;
    setIsCreatingNewChat(false);
    setError("");
    cancelEditUserMessage();
    cancelRenameSidebarChat();
    setChatMenu(null);
    try {
      const { data } = await api.get(`/chats/${chatId}/messages`);
      setMessages(
        normalizeMessagesFromApi(
          (data.messages || []).filter((message) => String(message?.content || "").trim().length > 0)
        )
      );
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to load this chat.");
    }
  };

  const exportChat = async (chat, format = "csv") => {
    if (!chat?.id) return;
    try {
      const { data } = await api.get(`/chats/${chat.id}/messages`);
      const rows = normalizeMessagesFromApi(
        (data.messages || []).filter((message) => String(message?.content || "").trim().length > 0)
      );
      const safeName = String(chat.title || `chat_${chat.id}`).replace(/[^a-z0-9]+/gi, "_").toLowerCase();
      let blob;
      let extension = "csv";

      if (format === "json") {
        extension = "json";
        const pairs = [];
        let pendingQuestion = null;
        for (const message of rows) {
          const role = String(message.role || "").toLowerCase();
          const content = String(message.content || "").trim();
          if (role === "user") {
            pendingQuestion = content;
          } else if (role === "assistant" && pendingQuestion != null) {
            const retrieval_context = Array.isArray(message.retrieval_context)
              ? message.retrieval_context.map((c) => String(c ?? "").trim()).filter((c) => c.length > 0)
              : [];
            pairs.push({
              input: pendingQuestion,
              actual_output: content,
              retrieval_context,
            });
            pendingQuestion = null;
          }
        }
        blob = new Blob([JSON.stringify(pairs, null, 2)], { type: "application/json;charset=utf-8" });
      } else if (format === "md") {
        extension = "md";
        const mdLines = [
          `# ${chat.title || `Chat ${chat.id}`}`,
          "",
          `- Chat ID: ${chat.id}`,
          `- Exported At: ${new Date().toISOString()}`,
          "",
          "---",
          "",
          ...rows.flatMap((message) => [
            `## ${String(message.role || "assistant").toUpperCase()}`,
            "",
            String(message.content || "").trim(),
            "",
            message.created_at ? `_${message.created_at}_` : "",
            "",
          ]),
        ];
        blob = new Blob([mdLines.join("\n")], { type: "text/markdown;charset=utf-8;" });
      } else {
        const escapeCsv = (value) => `"${String(value ?? "").replace(/"/g, '""')}"`;
        const csvLines = [
          ["chat_id", "chat_title", "role", "content", "created_at"].map(escapeCsv).join(","),
          ...rows.map((message) =>
            [chat.id, chat.title || "", message.role || "", message.content || "", message.created_at || ""]
              .map(escapeCsv)
              .join(",")
          ),
        ];
        blob = new Blob([csvLines.join("\n")], { type: "text/csv;charset=utf-8;" });
      }

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${safeName}.${extension}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      showSuccessToast(`Chat exported as ${extension.toUpperCase()}.`);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to export this chat.");
    }
  };

  const logout = () => {
    localStorage.clear();
    navigate("/login", { replace: true });
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(String(text || ""));
      showSuccessToast("Copied");
    } catch (_err) {
      setError("Unable to copy text. Please copy manually.");
    }
  };

  const closePdfViewer = useCallback(() => {
    if (pdfObjectUrlRef.current) {
      URL.revokeObjectURL(pdfObjectUrlRef.current);
      pdfObjectUrlRef.current = null;
    }
    setPdfObjectUrl(null);
    setShowPdfViewer(false);
    setPdfLoading(false);
  }, []);

  const openPdfViewer = async () => {
    if (!selectedDocument) return;
    const token = localStorage.getItem("token");
    if (!token) {
      setError("You are not signed in. Please log in again.");
      navigate("/login");
      return;
    }
    setPdfLoading(true);
    setError("");
    try {
      const base = String(api.defaults.baseURL || API_BASE_URL).replace(/\/$/, "");
      const pdfUrl = `${base}/files/${selectedDocument.id}/pdf`;
      const res = await fetch(pdfUrl, {
        headers: { Authorization: `Bearer ${token}` },
        credentials: "omit",
      });
      if (res.status === 401) {
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        setError("Session expired. Please sign in again.");
        navigate("/login");
        return;
      }
      if (!res.ok) {
        let msg = `Could not load PDF (${res.status}).`;
        try {
          const parsed = await res.json();
          if (parsed?.detail) msg = String(parsed.detail);
        } catch {
          /* keep default */
        }
        setError(msg);
        return;
      }
      const blob = await res.blob();
      if (!blob || blob.size === 0) {
        setError("Empty PDF response from server.");
        return;
      }
      const typed = blob.type === "application/pdf" ? blob : new Blob([blob], { type: "application/pdf" });
      if (pdfObjectUrlRef.current) {
        URL.revokeObjectURL(pdfObjectUrlRef.current);
      }
      const url = URL.createObjectURL(typed);
      pdfObjectUrlRef.current = url;
      setPdfObjectUrl(url);
      setShowPdfViewer(true);
    } catch (err) {
      setError(err?.message || "Could not load PDF.");
    } finally {
      setPdfLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (pdfObjectUrlRef.current) {
        URL.revokeObjectURL(pdfObjectUrlRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!showPdfViewer) return;
    const onKey = (e) => {
      if (e.key === "Escape") closePdfViewer();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showPdfViewer, closePdfViewer]);

  useEffect(() => {
    if (!showPdfViewer) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [showPdfViewer]);

  return (
    <div className="chat-shell">
      <div className="chat-shell-ambient" aria-hidden="true" />
      <aside className="sidebar">
        <div className="sidebar-top chatgpt-nav">
          <button className="nav-action primary" onClick={openNewChat}>
            + New chat
          </button>
        </div>

        <div className="sidebar-section-title">Documents</div>
        <div className="sidebar-top compact doc-actions">
          <select
            className="doc-select"
            value={selectedDocument?.id || ""}
            onChange={(e) => {
              const next = files.find((file) => file.id === Number(e.target.value)) || null;
              setSelectedDocument(next);
            }}
          >
            {!files.length ? <option value="">No documents</option> : null}
            {files.map((file) => (
              <option key={file.id} value={file.id}>
                {file.filename}
              </option>
            ))}
          </select>
          <div className={`doc-action-row${uploading ? " doc-action-row--uploading" : ""}`}>
            <label className="upload-btn doc-pill-btn" title="Upload a PDF from your device" aria-label="Upload PDF">
              {uploading ? (
                <span className="upload-progress-wrap">
                  <span className="upload-progress-label doc-upload-progress-head">
                    <DocIconUpload className="doc-action-icon--muted" />
                    <span>{`Uploading ${uploadProgress}%`}</span>
                  </span>
                  <span className="upload-progress-track">
                    <span className="upload-progress-fill" style={{ width: `${uploadProgress}%` }} />
                  </span>
                </span>
              ) : (
                <span className="doc-action-content">
                  <DocIconUpload />
                  <span className="doc-action-text">Upload</span>
                </span>
              )}
              <input type="file" accept="application/pdf" onChange={uploadFile} disabled={uploading} />
            </label>
            <button
              type="button"
              className="view-pdf-btn doc-pill-btn"
              onClick={openPdfViewer}
              disabled={!selectedDocument || pdfLoading || uploading}
              title={selectedDocument ? "Open this PDF in a preview window" : "Select a document first"}
              aria-label="View PDF"
            >
              {pdfLoading ? (
                <span className="doc-action-content">
                  <DocIconView className="doc-action-icon--pulse" />
                  <span className="doc-action-text">Opening…</span>
                </span>
              ) : (
                <span className="doc-action-content">
                  <DocIconView />
                  <span className="doc-action-text">View</span>
                </span>
              )}
            </button>
            <button
              type="button"
              className="delete-doc-btn doc-pill-btn"
              onClick={() => setShowDeleteModal(true)}
              disabled={!selectedDocument || deletingDocument || uploading}
              title="Remove this PDF and its chats"
              aria-label="Delete PDF"
            >
              {deletingDocument ? (
                <span className="doc-action-content">
                  <DocIconTrash className="doc-action-icon--pulse" />
                  <span className="doc-action-text">Deleting…</span>
                </span>
              ) : (
                <span className="doc-action-content">
                  <DocIconTrash />
                  <span className="doc-action-text">Delete</span>
                </span>
              )}
            </button>
          </div>
        </div>

        <div className="sidebar-section-title">Recents</div>
        <div
          className="file-list"
          ref={fileListRef}
          onScroll={() => {
            isUserScrollingSidebarRef.current = true;
            if (sidebarScrollAnimationRef.current) {
              cancelAnimationFrame(sidebarScrollAnimationRef.current);
              sidebarScrollAnimationRef.current = null;
              isSidebarScrollAnimatingRef.current = false;
            }
            if (sidebarScrollIdleTimerRef.current) {
              clearTimeout(sidebarScrollIdleTimerRef.current);
            }
            sidebarScrollIdleTimerRef.current = setTimeout(() => {
              isUserScrollingSidebarRef.current = false;
            }, 180);
          }}
        >
          {chats.map((chat) => (
            <div
              key={chat.id}
              ref={(node) => {
                if (node) {
                  chatItemRefs.current[chat.id] = node;
                } else {
                  delete chatItemRefs.current[chat.id];
                }
              }}
              className={[
                "file-item file-item--chat",
                selectedChatId === chat.id ? "active" : "",
                chatMenu?.chatId === chat.id ? "chat-item--overflow-open" : "",
              ]
                .filter(Boolean)
                .join(" ")}
            >
              {editingSidebarChatId === chat.id ? (
                <form className="chat-item-rename-form" onSubmit={submitRenameSidebarChat}>
                  <input
                    ref={renameChatInputRef}
                    className="chat-item-rename-input"
                    value={editingSidebarTitle}
                    onChange={(e) => setEditingSidebarTitle(e.target.value)}
                    maxLength={255}
                    disabled={renamingChat}
                    onKeyDown={(e) => {
                      if (e.key === "Escape") {
                        e.preventDefault();
                        if (!renamingChat) cancelRenameSidebarChat();
                      }
                    }}
                    aria-label="Conversation name"
                  />
                  <div className="chat-item-rename-actions">
                    <button type="submit" className="chat-item-rename-save" disabled={renamingChat}>
                      {renamingChat ? "Saving…" : "Save"}
                    </button>
                    <button
                      type="button"
                      className="chat-item-rename-cancel"
                      onClick={cancelRenameSidebarChat}
                      disabled={renamingChat}
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              ) : (
                <div className="chat-item-row">
                  <button type="button" className="chat-item-main" onClick={() => openExistingChat(chat.id)}>
                    <span className="chat-item-title-stack">
                      {isChatPinned(chat) ? (
                        <IconPin className="chat-item-inline-pin" aria-hidden />
                      ) : null}
                      <strong>{chat.title}</strong>
                    </span>
                  </button>
                  <div className="chat-item-overflow">
                    <button
                      type="button"
                      className="chat-item-more-btn"
                      data-chat-overflow-trigger
                      aria-expanded={chatMenu?.chatId === chat.id}
                      aria-haspopup="menu"
                      aria-label="Conversation options"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (uploading || sending || renamingChat || deletingChat) return;
                        if (chatMenu?.chatId === chat.id) {
                          setChatMenu(null);
                        } else {
                          const rect = e.currentTarget.getBoundingClientRect();
                          const width = 220;
                          setChatMenu({
                            chatId: chat.id,
                            top: rect.bottom + 6,
                            left: Math.min(window.innerWidth - width - 8, Math.max(8, rect.right - width)),
                            chat,
                          });
                        }
                      }}
                      disabled={uploading || sending || renamingChat || deletingChat}
                    >
                      <IconMoreHorizontal />
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))}
          {!selectedDocument ? (
            <div className="empty-state">Upload and select a PDF first.</div>
          ) : null}
          {selectedDocument && !chats.length ? <div className="empty-state">No chats yet. Start a new chat.</div> : null}
        </div>
        <div className="sidebar-footer">
          <div className="profile-chip">
            <div className="avatar-dot">{(user.name || user.email || "U").slice(0, 1).toUpperCase()}</div>
            <div>
              <strong>{user.name || "User"}</strong>
              <span>{user.email || ""}</span>
            </div>
          </div>
          <button className="logout-btn" onClick={logout}>
            Log out
          </button>
        </div>
      </aside>
      <main className="chat-main">
        <header className="chat-header">
          <div>
            <h3>{selectedDocument ? selectedDocument.filename : "Select a PDF"}</h3>
            <p className="chat-subtitle">
              {selectedDocument
                ? selectedChatId
                  ? "Chat history is loaded from MySQL."
                  : "New chat ready. Ask your first question."
                : "Choose a document to begin."}
            </p>
          </div>
          <div className="chat-header-actions">
            <select
              className="chat-export-format-select"
              value={exportFormat}
              onChange={(e) => {
                const selectedFormat = e.target.value;
                setExportFormat(selectedFormat);
                const currentChat = chats.find((chat) => chat.id === selectedChatId);
                if (currentChat && (selectedFormat === "csv" || selectedFormat === "md" || selectedFormat === "json")) {
                  exportChat(currentChat, selectedFormat);
                  setExportFormat("");
                }
              }}
              disabled={!selectedChatId}
            >
              <option value="">Export Chat</option>
              <option value="csv">Export as CSV</option>
              <option value="md">Export as MD</option>
              <option value="json">Export as JSON</option>
            </select>
          </div>
        </header>
        <section className="messages" ref={messagesContainerRef} onScroll={handleMessagesScroll}>
          {!messages.length ? (
            <div className="welcome-box">Ask anything about your selected PDF.</div>
          ) : (
            messages
              .filter(
                (message) =>
                  String(message?.content || "").trim().length > 0 || isStreamPlaceholderMessage(message)
              )
              .map((message, index) => (
              <div
                key={message.id || index}
                className={[
                  "message",
                  message.role,
                  message.role === "user" && editingUserMessageId === message.id ? "message-user--editing" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
              >
                <div
                  className={[
                    "bubble-wrap",
                    message.role === "user" ? "bubble-wrap--user-hover" : "",
                    message.role === "user" && editingUserMessageId === message.id ? "bubble-wrap--user-editing" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                >
                  {message.role === "assistant" && String(message.content || "").trim().length > 0 ? (
                    <button
                      type="button"
                      className="copy-answer-btn"
                      onClick={() => copyToClipboard(message.content)}
                      aria-label="Copy answer"
                      title="Copy"
                    >
                      <IconCopy className="copy-answer-btn-icon" />
                    </button>
                  ) : null}
                  {message.role === "user" &&
                  message.id != null &&
                  Number.isFinite(Number(message.id)) &&
                  selectedChatId &&
                  !sending &&
                  editingUserMessageId !== message.id ? (
                    <button
                      type="button"
                      className="message-edit-btn"
                      onClick={() => {
                        setEditingUserMessageId(message.id);
                        setEditingDraft(String(message.content || ""));
                      }}
                      aria-label="Edit message"
                      title="Edit question"
                    >
                      <DocIconEdit />
                    </button>
                  ) : null}
                  {message.role === "user" && editingUserMessageId === message.id ? (
                    <form
                      className="user-message-editor"
                      onSubmit={(e) => {
                        e.preventDefault();
                        submitEditedUserMessage(message.id);
                      }}
                    >
                      <div className="user-message-editor-row">
                        <textarea
                          ref={userMessageEditInputRef}
                          className="user-message-editor-input"
                          value={editingDraft}
                          onChange={(e) => setEditingDraft(e.target.value)}
                          rows={2}
                          autoFocus
                          aria-label="Edit your question"
                          onKeyDown={(e) => {
                            if (e.key === "Escape") {
                              e.preventDefault();
                              cancelEditUserMessage();
                              return;
                            }
                            if (e.key === "Enter" && !e.shiftKey) {
                              e.preventDefault();
                              submitEditedUserMessage(message.id);
                            }
                          }}
                        />
                        <div className="user-message-editor-toolbar" role="group" aria-label="Edit actions">
                          <button
                            type="submit"
                            className="user-message-icon-btn user-message-icon-btn--submit"
                            title="Save and re-run (Enter). Shift+Enter for a new line."
                            aria-label="Save and re-run"
                          >
                            <SendPromptIcon className="user-message-editor-toolbar-icon" />
                          </button>
                          <button
                            type="button"
                            className="user-message-icon-btn user-message-icon-btn--cancel"
                            onClick={cancelEditUserMessage}
                            title="Cancel editing"
                            aria-label="Cancel editing"
                          >
                            <IconClose className="user-message-editor-toolbar-icon" />
                          </button>
                        </div>
                      </div>
                    </form>
                  ) : message.role === "assistant" &&
                    isStreamPlaceholderMessage(message) &&
                    !String(message.content || "").trim() ? (
                    <div className="bubble typing-bubble">
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                    </div>
                  ) : message.role === "assistant" ? (
                    <AssistantMarkdown content={message.content} />
                  ) : (
                    <div className="bubble">{message.content}</div>
                  )}
                  {message.role === "assistant" && (() => {
                    const metaLine = formatAssistantMetrics(message);
                    return metaLine ? <div className="message-meta">{metaLine}</div> : null;
                  })()}
                  {message.role === "assistant" && Array.isArray(message.source_citations) && message.source_citations.length ? (
                    <div className="message-meta">
                      {message.source_citations.slice(0, 3).map((src, i) => {
                        const page = src?.page != null ? `p.${src.page}` : "source";
                        const section = src?.section ? ` ${src.section}` : "";
                        return (
                          <div key={`${message.id || index}-src-${i}`} title={String(src?.snippet || "")}>
                            [{i + 1}] {page}{section}
                          </div>
                        );
                      })}
                    </div>
                  ) : null}
                  {message.role === "assistant" && Number.isFinite(Number(message.id)) ? (
                    <div className="message-meta">
                      <button
                        type="button"
                        className={`copy-answer-btn${Number(message.feedback_rating) === 1 ? " active" : ""}`}
                        onClick={() => submitAssistantFeedback(message.id, 1)}
                        aria-label="Mark answer helpful"
                        title="Helpful"
                      >
                        <IconThumbUp className="copy-answer-btn-icon" />
                      </button>
                      <button
                        type="button"
                        className={`copy-answer-btn${Number(message.feedback_rating) === -1 ? " active" : ""}`}
                        onClick={() => submitAssistantFeedback(message.id, -1)}
                        aria-label="Mark answer not helpful"
                        title="Not helpful"
                      >
                        <IconThumbDown className="copy-answer-btn-icon" />
                      </button>
                    </div>
                  ) : null}
                </div>
              </div>
              ))
          )}
          {sending && !messages.some(isStreamPlaceholderMessage) ? (
            <div className="message assistant">
              <div className="bubble typing-bubble">
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="typing-dot" />
              </div>
            </div>
          ) : null}
          <div ref={messagesEndRef} />
        </section>
        {showScrollToLatest ? (
          <button
            type="button"
            className="scroll-to-latest"
            onClick={jumpToLatest}
            aria-label="Scroll to latest message"
          >
            <span className="scroll-to-latest-arrow" aria-hidden="true">↓</span>
            Latest
          </button>
        ) : null}
        <form className="composer" onSubmit={sendQuestion}>
          <textarea
            ref={questionInputRef}
            className="composer-input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendQuestion(e);
              }
            }}
            placeholder="Ask a question…  (Enter to send · Shift+Enter for a new line)"
            rows={1}
            disabled={!selectedDocument || sending}
            aria-label="Ask a question about your PDF"
          />
          {sending ? (
            <button type="button" className="stop-btn" onClick={stopProcess} aria-label="Stop process" title="Stop">
              <IconStop className="stop-btn-icon" />
            </button>
          ) : (
            <button
              type="submit"
              className="composer-send-btn"
              disabled={!selectedDocument || sending}
              aria-label="Send prompt"
              title="Send prompt (Enter)"
            >
              <SendPromptIcon className="composer-send-icon" />
            </button>
          )}
        </form>
        {error && (
          <div className="error-text chat-error" role="alert">
            <span className="chat-error-text">{error}</span>
            <button
              type="button"
              className="chat-error-dismiss"
              onClick={() => setError("")}
              aria-label="Dismiss error"
            >
              <IconClose className="chat-error-dismiss-icon" />
            </button>
          </div>
        )}
      </main>
      {showPdfViewer && pdfObjectUrl && selectedDocument ? (
        <div
          className="pdf-viewer-backdrop"
          onClick={closePdfViewer}
          role="presentation"
        >
          <div
            className="pdf-viewer-shell"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="pdf-viewer-title"
          >
            <header className="pdf-viewer-toolbar">
              <div className="pdf-viewer-title-block">
                <span className="pdf-viewer-badge" aria-hidden="true">
                  PDF
                </span>
                <h2 id="pdf-viewer-title" className="pdf-viewer-title">
                  {selectedDocument.filename}
                </h2>
              </div>
              <button type="button" className="pdf-viewer-close" onClick={closePdfViewer} aria-label="Close viewer">
                ×
              </button>
            </header>
            <div className="pdf-viewer-frame-wrap">
              <iframe title={selectedDocument.filename} src={pdfObjectUrl} className="pdf-viewer-frame" />
            </div>
          </div>
        </div>
      ) : null}
      {showDeleteModal && selectedDocument ? (
        <div className="modal-backdrop" onClick={() => !deletingDocument && setShowDeleteModal(false)}>
          <div className="confirm-modal" onClick={(e) => e.stopPropagation()}>
            <h4>Delete PDF?</h4>
            <p>
              Delete <strong>{selectedDocument.filename}</strong>? This will also remove all chats for this file.
            </p>
            <div className="confirm-modal-actions">
              <button
                type="button"
                className="modal-cancel-btn"
                onClick={() => setShowDeleteModal(false)}
                disabled={deletingDocument}
              >
                Cancel
              </button>
              <button type="button" className="modal-delete-btn" onClick={deleteSelectedDocument} disabled={deletingDocument}>
                {deletingDocument ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {deleteChatTarget ? (
        <div className="modal-backdrop" onClick={() => !deletingChat && setDeleteChatTarget(null)}>
          <div className="confirm-modal" onClick={(e) => e.stopPropagation()}>
            <h4>Remove conversation?</h4>
            <p>
              Remove <strong>{deleteChatTarget.title || "this conversation"}</strong> from Recents? It will be hidden
              from your list; messages stay stored on the server.
            </p>
            <div className="confirm-modal-actions">
              <button
                type="button"
                className="modal-cancel-btn"
                onClick={() => setDeleteChatTarget(null)}
                disabled={deletingChat}
              >
                Cancel
              </button>
              <button type="button" className="modal-delete-btn" onClick={confirmSoftDeleteChat} disabled={deletingChat}>
                {deletingChat ? "Removing…" : "Remove"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {chatMenu
        ? createPortal(
            <div
              ref={chatOverflowMenuRef}
              className="chat-overflow-menu"
              data-chat-overflow="menu"
              style={{
                position: "fixed",
                top: chatMenu.top,
                left: chatMenu.left,
                zIndex: 200,
              }}
              role="menu"
            >
              <button
                type="button"
                role="menuitem"
                className="chat-overflow-menu-item"
                onClick={() => {
                  const c = chatMenu.chat;
                  setChatMenu(null);
                  beginRenameSidebarChat(c);
                }}
              >
                <DocIconEdit className="chat-overflow-menu-icon" />
                <span>Rename</span>
              </button>
              <button
                type="button"
                role="menuitem"
                className="chat-overflow-menu-item"
                onClick={() => handleTogglePinChat(chatMenu.chat)}
              >
                <IconPin className="chat-overflow-menu-icon" />
                <span>{isChatPinned(chatMenu.chat) ? "Unpin chat" : "Pin chat"}</span>
              </button>
              <button
                type="button"
                role="menuitem"
                className="chat-overflow-menu-item"
                onClick={() => handleArchiveChat(chatMenu.chat)}
              >
                <IconArchive className="chat-overflow-menu-icon" />
                <span>Archive</span>
              </button>
              <button
                type="button"
                role="menuitem"
                className="chat-overflow-menu-item chat-overflow-menu-item--danger"
                onClick={() => {
                  setDeleteChatTarget(chatMenu.chat);
                  setChatMenu(null);
                }}
              >
                <DocIconTrash className="chat-overflow-menu-icon" />
                <span>Delete</span>
              </button>
            </div>,
            document.body
          )
        : null}
      {toastMessage ? (
        <div className="app-toast success" role="status" aria-live="polite">
          <span className="app-toast-icon" aria-hidden="true">
            ✓
          </span>
          <div className="app-toast-body">
            <strong>Success</strong>
            <span>{toastMessage}</span>
          </div>
        </div>
      ) : null}
    </div>
  );
}
