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

// 从问题里抓"首驱某车型"，留资表单自动预填"想看的车型"
function detectModel(q) {
  const m = q.match(/首驱\s*([A-Za-z0-9][A-Za-z0-9\- ]*)/)
  if (m && m[1].trim()) return '首驱' + m[1].trim()
  return ''
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
    // 快捷按钮：对比 / 到店 / 预约看车（带 action 的不发问，直接弹层）
    suggestions: [
      { q: '雅迪和首驱哪个好？', label: '⚡ 和雅迪比' },
      { q: '九号和首驱怎么选？', label: '⚡ 和九号比' },
      { q: '首驱电动车有什么卖点？', label: '💡 首驱卖点' },
      { q: '首驱S300和九号E300P比哪个强？', label: '🏆 旗舰对比' },
      { action: 'store', label: '🏪 到店' },
      { action: 'lead', label: '📱 预约看车' },
    ],
    // 门店信息 + 留资
    showStore: false,
    showLead: false,
    leadPhone: '',
    leadModel: '',
    leadSaving: false,
    store: {},
  },

  onLoad() {
    // 拉门店信息（地址/电话/营业时间），用于"到店"弹层
    wx.request({
      url: API + '/api/store',
      success: (res) => {
        const s = (res.data && res.data.store) || {}
        if (s.name) this.setData({ store: s })
      },
    })
  },

  noop() {},

  onInput(e) {
    this.setData({ draft: e.detail.value })
  },

  onSuggest(e) {
    const { q, action } = e.currentTarget.dataset
    if (action === 'store') { this.showStore(); return }
    if (action === 'lead') { this.showLead(); return }
    const qq = (q || '').trim()
    if (!qq || this.data.loading) return
    this.setData({ draft: qq })
    this.onSend()
  },

  onSend() {
    const q = (this.data.draft || '').trim()
    if (!q || this.data.loading) return

    const messages = this.data.messages.concat([{ role: 'user', content: q, time: nowTime() }])
    this.setData({ messages, draft: '', loading: true })
    this.lastModel = detectModel(q) || this.lastModel || ''

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

  // —— 到店 / 留资（P0：邀约到店 + 抓潜在客户）——
  showStore() {
    if (this.data.store.name) {
      this.setData({ showStore: true })
    } else {
      wx.request({
        url: API + '/api/store',
        success: (res) => {
          const s = (res.data && res.data.store) || {}
          if (s.name) this.setData({ store: s, showStore: true })
          else wx.showToast({ title: '门店信息还没填，老板去 data/store.json 填', icon: 'none' })
        },
      })
    }
  },
  hideStore() { this.setData({ showStore: false }) },
  callStore() {
    const phone = this.data.store.phone
    if (!phone || phone.indexOf('0000000') >= 0) {
      wx.showToast({ title: '门店电话还没填', icon: 'none' })
      return
    }
    wx.makePhoneCall({ phoneNumber: phone.replace(/-/g, '') })
  },
  navigateStore() {
    const s = this.data.store
    if (!s.lat || !s.lng) {
      wx.showToast({ title: '门店位置还没填', icon: 'none' })
      return
    }
    wx.openLocation({ latitude: s.lat, longitude: s.lng, name: s.name, address: s.address })
  },

  showLead() {
    this.setData({ leadModel: this.lastModel || '', showLead: true })
  },
  hideLead() { this.setData({ showLead: false }) },
  onLeadPhone(e) { this.setData({ leadPhone: e.detail.value }) },
  onLeadModel(e) { this.setData({ leadModel: e.detail.value }) },
  submitLead() {
    const phone = (this.data.leadPhone || '').trim()
    if (!/^1\d{10}$/.test(phone)) {
      wx.showToast({ title: '手机号要 11 位、1 开头', icon: 'none' })
      return
    }
    this.setData({ leadSaving: true })
    wx.request({
      url: API + '/api/lead',
      method: 'POST',
      header: { 'Content-Type': 'application/json' },
      data: { phone, model: (this.data.leadModel || '').trim() },
      success: (res) => {
        if (res.data && res.data.ok) {
          this.setData({ showLead: false, leadPhone: '', leadSaving: false })
          wx.showModal({
            title: '已登记 ✅',
            content: '门店会尽快联系您，也可以直接到店看车！',
            showCancel: false,
            confirmText: '好的',
          })
        } else {
          wx.showToast({ title: (res.data && res.data.error) || '提交失败', icon: 'none' })
          this.setData({ leadSaving: false })
        }
      },
      fail: () => {
        wx.showToast({ title: '连不上后端', icon: 'none' })
        this.setData({ leadSaving: false })
      },
    })
  },
})
