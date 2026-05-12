import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

export default function ChatPage() {
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
  const messagesEndRef = useRef(null);
  const toastTimerRef = useRef(null);
  const streamAbortRef = useRef(null);
  const navigate = useNavigate();
  const user = useMemo(() => JSON.parse(localStorage.getItem("user") || "{}"), []);

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
        setMessages(messagesResp.data.messages || []);
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

  useEffect(() => {
    animateScrollToBottom(true);
  }, [messages, selectedChatId]);

  useEffect(() => {
    // Keep newest user question / streamed answer visible.
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, sending]);

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

  const sendQuestion = async (event) => {
    event.preventDefault();
    if (!selectedDocument || !question.trim() || sending) return;
    const userMessage = { role: "user", content: question.trim() };
    const streamMessageId = `stream-${Date.now()}`;
    setMessages((prev) => [...prev, userMessage, { id: streamMessageId, role: "assistant", content: "" }]);
    const startedAt = performance.now();
    const messageText = question.trim();
    let streamText = "";
    let activeChatId = selectedChatIdRef.current;
    setQuestion("");
    setSending(true);
    setError("");
    try {
      const controller = new AbortController();
      streamAbortRef.current = controller;
      const token = localStorage.getItem("token");
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: "POST",
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          Authorization: token ? `Bearer ${token}` : "",
        },
        body: JSON.stringify({
          document_id: selectedDocument.id,
          question: messageText,
          chat_id: isCreatingNewChat ? null : selectedChatIdRef.current,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error("Failed to start streaming response.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let donePayload = null;

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
            if (payload.chat_id) {
              activeChatId = payload.chat_id;
            }
          } else if (payload.type === "done") {
            donePayload = payload;
            if (payload.chat_id) {
              activeChatId = payload.chat_id;
            }
            const elapsedSeconds = Number(((performance.now() - startedAt) / 1000).toFixed(2));
            const accuracy = calculateAccuracyFromDistances(payload.distances || []);
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === streamMessageId
                  ? {
                      ...msg,
                      content: payload.answer || msg.content,
                      latency_seconds: elapsedSeconds,
                      accuracy_label: accuracy.label,
                      accuracy_score: accuracy.score,
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
      }
    } catch (err) {
      if (err?.name === "AbortError") {
        const stoppedContent = streamText.trim() || "Response stopped.";
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === streamMessageId
              ? { ...msg, content: stoppedContent, stopped: true }
              : msg
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
          } catch (_saveErr) {
            // Non-fatal: user still sees stopped content in UI.
          }
        }
        showSuccessToast("Stopped");
      } else {
        setMessages((prev) => prev.filter((msg) => msg.id !== streamMessageId));
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
    try {
      const { data } = await api.get(`/chats/${chatId}/messages`);
      setMessages((data.messages || []).filter((message) => String(message?.content || "").trim().length > 0));
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to load this chat.");
    }
  };

  const exportChat = async (chat, format = "csv") => {
    if (!chat?.id) return;
    try {
      const { data } = await api.get(`/chats/${chat.id}/messages`);
      const rows = (data.messages || []).filter((message) => String(message?.content || "").trim().length > 0);
      const safeName = String(chat.title || `chat_${chat.id}`).replace(/[^a-z0-9]+/gi, "_").toLowerCase();
      let blob;
      let extension = "csv";

      if (format === "md") {
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
    navigate("/login");
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(String(text || ""));
      showSuccessToast("Copied");
    } catch (_err) {
      setError("Unable to copy text. Please copy manually.");
    }
  };

  return (
    <div className="chat-shell">
      <aside className="sidebar">
        <div className="sidebar-top chatgpt-nav">
          <button className="nav-action primary" onClick={openNewChat}>
            + New chat
          </button>
        </div>

        <div className="sidebar-section-title">Documents</div>
        <div className="sidebar-top compact">
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
          <label className="upload-btn">
            {uploading ? (
              <span className="upload-progress-wrap">
                <span className="upload-progress-label">{`Uploading ${uploadProgress}%`}</span>
                <span className="upload-progress-track">
                  <span className="upload-progress-fill" style={{ width: `${uploadProgress}%` }} />
                </span>
              </span>
            ) : (
              "Upload PDF"
            )}
            <input type="file" accept="application/pdf" onChange={uploadFile} disabled={uploading} />
          </label>
          <button
            type="button"
            className="delete-doc-btn"
            onClick={() => setShowDeleteModal(true)}
            disabled={!selectedDocument || deletingDocument || uploading}
          >
            {deletingDocument ? "Deleting..." : "Delete PDF"}
          </button>
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
              className={`file-item ${selectedChatId === chat.id ? "active" : ""}`}
            >
              <button type="button" className="chat-item-main" onClick={() => openExistingChat(chat.id)}>
                <strong>{chat.title}</strong>
              </button>
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
                if (currentChat && (selectedFormat === "csv" || selectedFormat === "md")) {
                  exportChat(currentChat, selectedFormat);
                  setExportFormat("");
                }
              }}
              disabled={!selectedChatId}
            >
              <option value="">Export Chat</option>
              <option value="csv">Export as CSV</option>
              <option value="md">Export as MD</option>
            </select>
          </div>
        </header>
        <section className="messages" ref={messagesContainerRef}>
          {!messages.length ? (
            <div className="welcome-box">Ask anything about your selected PDF.</div>
          ) : (
            messages
              .filter((message) => String(message?.content || "").trim().length > 0)
              .map((message, index) => (
              <div key={message.id || index} className={`message ${message.role}`}>
                <div className="bubble-wrap">
                  {message.role === "assistant" ? (
                    <button
                      type="button"
                      className="copy-answer-btn"
                      onClick={() => copyToClipboard(message.content)}
                      aria-label="Copy answer"
                      title="Copy"
                    >
                      📋
                    </button>
                  ) : null}
                  <div className="bubble">{message.content}</div>
                  {message.role === "assistant" && Number.isFinite(message.latency_seconds) ? (
                    <div className="message-meta">
                      {message.accuracy_label && Number.isFinite(message.accuracy_score)
                        ? `Accuracy: ${message.accuracy_label} (${message.accuracy_score}%) • `
                        : ""}
                      {`${message.latency_seconds}s`}
                    </div>
                  ) : null}
                </div>
              </div>
              ))
          )}
          {sending ? (
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
        <form className="composer" onSubmit={sendQuestion}>
          <input
            ref={questionInputRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question..."
            disabled={!selectedDocument || sending}
          />
          {sending ? (
            <button type="button" className="stop-btn" onClick={stopProcess} aria-label="Stop process" title="Stop">
              ⏹
            </button>
          ) : (
            <button type="submit" disabled={!selectedDocument || sending}>
              Send
            </button>
          )}
        </form>
        {error && <div className="error-text chat-error">{error}</div>}
      </main>
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
