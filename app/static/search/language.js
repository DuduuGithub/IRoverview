// 全局语言控制模块
document.addEventListener('DOMContentLoaded', function() {
    // 获取语言开关元素
    const languageSwitch = document.getElementById('languageSwitch');
    
    // 如果存在语言开关，添加事件监听
    if (languageSwitch) {
        // 从localStorage中获取当前语言设置，默认为中文
        const currentLanguage = localStorage.getItem('preferredLanguage') || 'zh';
        
        // 根据保存的设置更新UI
        updateLanguageUI(currentLanguage);
        
        // 添加语言切换事件
        languageSwitch.addEventListener('click', function() {
            // 获取当前语言
            const currentLanguage = localStorage.getItem('preferredLanguage') || 'zh';
            
            // 切换语言
            const newLanguage = currentLanguage === 'zh' ? 'en' : 'zh';
            
            // 保存新的语言设置
            localStorage.setItem('preferredLanguage', newLanguage);
            
            // 更新UI
            updateLanguageUI(newLanguage);
        });
    }
});

// 更新UI中的语言显示
function updateLanguageUI(language) {
    // 所有中文元素
    document.querySelectorAll('.text-zh').forEach(el => {
        el.style.display = language === 'zh' ? 'inline-block' : 'none';
    });
    
    // 所有英文元素
    document.querySelectorAll('.text-en').forEach(el => {
        el.style.display = language === 'en' ? 'inline-block' : 'none';
    });
} 