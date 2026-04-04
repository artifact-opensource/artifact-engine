/*
 * Artifact Engine — HTTP Model Downloader
 *
 * Downloads GGUF model files from HTTP URLs using raw sockets.
 * Shows progress bar with speed and ETA.
 * Supports resume via Content-Range.
 *
 * No external dependencies — uses Winsock on Windows, POSIX sockets on Linux.
 */

#include "../include/model_fetch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  typedef int socklen_t;
  #define CLOSESOCK closesocket
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netdb.h>
  #include <unistd.h>
  #include <arpa/inet.h>
  #define CLOSESOCK close
  typedef int SOCKET;
  #define INVALID_SOCKET -1
  #define SOCKET_ERROR -1
#endif

/* ───── URL parsing ───── */

typedef struct {
    char host[256];
    char path[1024];
    uint16_t port;
} parsed_url;

static bool parse_url(const char* url, parsed_url* out) {
    memset(out, 0, sizeof(*out));
    out->port = 80;
    
    const char* p = url;
    if (strncmp(p, "http://", 7) == 0) p += 7;
    else if (strncmp(p, "https://", 8) == 0) {
        /* We don't support HTTPS in this minimal client */
        fprintf(stderr, "fetch: HTTPS not supported, use HTTP\n");
        return false;
    }
    
    /* Extract host */
    const char* slash = strchr(p, '/');
    const char* colon = strchr(p, ':');
    
    if (colon && (!slash || colon < slash)) {
        /* host:port */
        size_t hlen = colon - p;
        if (hlen >= sizeof(out->host)) return false;
        memcpy(out->host, p, hlen);
        out->host[hlen] = 0;
        out->port = (uint16_t)atoi(colon + 1);
    } else if (slash) {
        size_t hlen = slash - p;
        if (hlen >= sizeof(out->host)) return false;
        memcpy(out->host, p, hlen);
        out->host[hlen] = 0;
    } else {
        strncpy(out->host, p, sizeof(out->host) - 1);
        strcpy(out->path, "/");
        return true;
    }
    
    if (slash) {
        strncpy(out->path, slash, sizeof(out->path) - 1);
    } else {
        strcpy(out->path, "/");
    }
    
    return true;
}

/* ───── Progress display ───── */

static void print_progress(size_t downloaded, size_t total, time_t start) {
    time_t now = time(NULL);
    double elapsed = difftime(now, start);
    if (elapsed < 0.5) elapsed = 0.5;
    
    double speed = downloaded / elapsed;
    double pct = total > 0 ? (100.0 * downloaded / total) : 0;
    
    double remaining = 0;
    if (speed > 0 && total > 0) {
        remaining = (total - downloaded) / speed;
    }
    
    char speed_str[32];
    if (speed > 1024 * 1024) snprintf(speed_str, sizeof(speed_str), "%.1f MB/s", speed / (1024*1024));
    else if (speed > 1024) snprintf(speed_str, sizeof(speed_str), "%.1f KB/s", speed / 1024);
    else snprintf(speed_str, sizeof(speed_str), "%.0f B/s", speed);
    
    /* Progress bar */
    int bar_width = 30;
    int filled = (int)(bar_width * pct / 100);
    
    printf("\r  [");
    for (int i = 0; i < bar_width; i++) {
        printf(i < filled ? "█" : "░");
    }
    printf("] %5.1f%% | %.2f/%.2f GB | %s | ETA %dm%02ds  ",
           pct,
           downloaded / (1024.0*1024*1024),
           total / (1024.0*1024*1024),
           speed_str,
           (int)(remaining / 60),
           (int)remaining % 60);
    fflush(stdout);
}

/* ───── Socket connect ───── */

static SOCKET tcp_connect(const char* host, uint16_t port) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    struct addrinfo hints = {0}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    
    char port_str[8];
    snprintf(port_str, sizeof(port_str), "%u", port);
    
    if (getaddrinfo(host, port_str, &hints, &res) != 0) {
        fprintf(stderr, "fetch: cannot resolve '%s'\n", host);
        return INVALID_SOCKET;
    }
    
    SOCKET sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock == INVALID_SOCKET) {
        freeaddrinfo(res);
        return INVALID_SOCKET;
    }
    
    if (connect(sock, res->ai_addr, (int)res->ai_addrlen) == SOCKET_ERROR) {
        fprintf(stderr, "fetch: cannot connect to %s:%u\n", host, port);
        CLOSESOCK(sock);
        freeaddrinfo(res);
        return INVALID_SOCKET;
    }
    
    freeaddrinfo(res);
    return sock;
}

/* ───── HTTP GET with streaming to file ───── */

bool model_fetch(const char* url, const char* output_path) {
    parsed_url pu;
    if (!parse_url(url, &pu)) return false;
    
    printf("fetch: downloading from %s:%u%s\n", pu.host, pu.port, pu.path);
    printf("fetch: saving to %s\n", output_path);
    
    /* Check for existing partial download */
    size_t existing = 0;
    FILE* fp = fopen(output_path, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        existing = (size_t)ftell(fp);
        fclose(fp);
        if (existing > 0) {
            printf("fetch: resuming from %.2f GB\n", existing / (1024.0*1024*1024));
        }
    }
    
    SOCKET sock = tcp_connect(pu.host, pu.port);
    if (sock == INVALID_SOCKET) return false;
    
    /* Send HTTP GET with Range header for resume */
    char request[2048];
    if (existing > 0) {
        snprintf(request, sizeof(request),
            "GET %s HTTP/1.1\r\n"
            "Host: %s:%u\r\n"
            "Range: bytes=%zu-\r\n"
            "Connection: close\r\n"
            "\r\n", pu.path, pu.host, pu.port, existing);
    } else {
        snprintf(request, sizeof(request),
            "GET %s HTTP/1.1\r\n"
            "Host: %s:%u\r\n"
            "Connection: close\r\n"
            "\r\n", pu.path, pu.host, pu.port);
    }
    
    send(sock, request, (int)strlen(request), 0);
    
    /* Read HTTP headers */
    char header_buf[8192];
    int header_len = 0;
    int status_code = 0;
    size_t content_length = 0;
    
    /* Read until we find \r\n\r\n */
    while (header_len < (int)sizeof(header_buf) - 1) {
        int n = recv(sock, header_buf + header_len, 1, 0);
        if (n <= 0) break;
        header_len += n;
        header_buf[header_len] = 0;
        if (header_len >= 4 && 
            memcmp(header_buf + header_len - 4, "\r\n\r\n", 4) == 0) break;
    }
    
    /* Parse status code */
    if (sscanf(header_buf, "HTTP/%*d.%*d %d", &status_code) != 1) {
        fprintf(stderr, "fetch: invalid HTTP response\n");
        CLOSESOCK(sock);
        return false;
    }
    
    if (status_code != 200 && status_code != 206) {
        fprintf(stderr, "fetch: HTTP %d\n", status_code);
        CLOSESOCK(sock);
        return false;
    }
    
    /* Parse Content-Length */
    const char* cl = strstr(header_buf, "Content-Length:");
    if (!cl) cl = strstr(header_buf, "content-length:");
    if (cl) {
        content_length = (size_t)strtoull(cl + 15, NULL, 10);
    }
    
    size_t total = content_length + existing;
    printf("fetch: %.2f GB to download (%.2f GB total)\n",
           content_length / (1024.0*1024*1024), total / (1024.0*1024*1024));
    
    /* Open file for writing (append if resuming) */
    fp = fopen(output_path, existing > 0 ? "ab" : "wb");
    if (!fp) {
        fprintf(stderr, "fetch: cannot open '%s' for writing\n", output_path);
        CLOSESOCK(sock);
        return false;
    }
    
    /* Stream body to file */
    char buf[65536];
    size_t downloaded = 0;
    time_t start_time = time(NULL);
    time_t last_progress = 0;
    
    while (1) {
        int n = recv(sock, buf, sizeof(buf), 0);
        if (n <= 0) break;
        
        fwrite(buf, 1, n, fp);
        downloaded += n;
        
        /* Progress every second */
        time_t now = time(NULL);
        if (now != last_progress) {
            print_progress(existing + downloaded, total, start_time);
            last_progress = now;
        }
    }
    
    printf("\n");
    fclose(fp);
    CLOSESOCK(sock);
    
#ifdef _WIN32
    WSACleanup();
#endif
    
    if (content_length > 0 && downloaded < content_length) {
        fprintf(stderr, "fetch: incomplete download (%zu/%zu bytes)\n", 
                downloaded, content_length);
        fprintf(stderr, "fetch: run again to resume\n");
        return false;
    }
    
    printf("fetch: done — %.2f GB downloaded\n", downloaded / (1024.0*1024*1024));
    return true;
}

const char* model_fetch_filename(const char* url) {
    const char* last_slash = strrchr(url, '/');
    return last_slash ? last_slash + 1 : url;
}
