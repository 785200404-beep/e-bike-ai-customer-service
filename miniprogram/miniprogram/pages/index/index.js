// 后端地址：
//   电脑上跑 `python server.py` 后，开发者工具里用 127.0.0.1 即可。
//   真机预览 / 分享给同事时，改成电脑的局域网 IP（手机和电脑要连同一个 Wi-Fi）。
//   查 IP：`ipconfig getifaddr en0`（Mac）
const API = 'http://192.168.1.6:8000'
const SID_KEY = 'cs_session_id'

// 会话 id：生成一次存在本地，之后每次都带同一个 → 客服能"记得"这一场对话
function getSessionId() {
  let sid = wx.getStorageSync(SID_KEY)
  if (!sid) {
    sid = Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8)
    wx.setStorageSync(SID_KEY, sid)
  }
  return sid
}

Page({
  data: {
    messages: [
      { role: 'bot', content: '您好，我是首驱电动车店客服。可以问我价格、续航、库存、算账，比如「首驱Sz110多少钱？」' }
    ],
    draft: '',
    loading: false,
  },

  onInput(e) {
    this.setData({ draft: e.detail.value })
  },

  onSend() {
    const q = (this.data.draft || '').trim()
    if (!q || this.data.loading) return

    const messages = this.data.messages.concat([{ role: 'user', content: q }])
    this.setData({ messages, draft: '', loading: true })

    wx.request({
      url: API + '/api/chat',
      method: 'POST',
      header: { 'Content-Type': 'application/json' },
      data: { question: q, session_id: getSessionId() },
      success: (res) => {
        const data = res.data || {}
        let answer = data.answer || ''
        const tools = data.used_tools || []
        if (tools.length) {
          const names = tools.map(t => (t.includes('计算器') ? '🧮 算账' : '📦 查库存')).join('、')
          answer += '\n\n（已用工具：' + names + '）'
        }
        if (!answer) answer = '（服务返回异常，请检查后端日志）'
        this.setData({ messages: this.data.messages.concat([{ role: 'bot', content: answer }]) })
      },
      fail: () => {
        this.setData({
          messages: this.data.messages.concat([{
            role: 'bot',
            content: '连不上后端。请先跑：python server.py（开发者工具还要勾选「不校验合法域名」）'
          }])
        })
      },
      complete: () => this.setData({ loading: false }),
    })
  },
})
