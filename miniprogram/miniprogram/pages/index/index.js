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

// 当前时间 HH:MM，显示在每条消息下面
function nowTime() {
  const d = new Date()
  const p = n => (n < 10 ? '0' + n : '' + n)
  return p(d.getHours()) + ':' + p(d.getMinutes())
}

Page({
  data: {
    messages: [
      {
        role: 'bot',
        welcome: true,
        content: '📌 首驱电动车 · 南宁一网\n营业时间：每天 9:00-21:00，周末不休\n以旧换新：旧车抵 300-800 元 ｜ 代办上牌 50 元\n可问：价格、续航、库存、算账、售后、竞品对比',
        time: nowTime()
      }
    ],
    draft: '',
    loading: false,
    // 快捷对比按钮：点一下直接发问
    suggestions: [
      { q: '雅迪和首驱哪个好？', label: '⚡ 和雅迪比' },
      { q: '九号和首驱怎么选？', label: '⚡ 和九号比' },
      { q: '首驱电动车有什么卖点？', label: '💡 首驱卖点' },
      { q: '首驱S300和九号E300P比哪个强？', label: '🏆 旗舰对比' },
    ],
  },

  onInput(e) {
    this.setData({ draft: e.detail.value })
  },

  onSuggest(e) {
    const q = (e.currentTarget.dataset.q || '').trim()
    if (!q || this.data.loading) return
    this.setData({ draft: q })
    this.onSend()
  },

  onSend() {
    const q = (this.data.draft || '').trim()
    if (!q || this.data.loading) return

    const messages = this.data.messages.concat([{ role: 'user', content: q, time: nowTime() }])
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
        this.setData({
          messages: this.data.messages.concat([{ role: 'bot', content: answer, time: nowTime() }])
        })
      },
      fail: () => {
        this.setData({
          messages: this.data.messages.concat([{
            role: 'bot',
            content: '连不上后端。请先跑：python server.py（开发者工具还要勾选「不校验合法域名」）',
            time: nowTime()
          }])
        })
      },
      complete: () => this.setData({ loading: false }),
    })
  },
})
